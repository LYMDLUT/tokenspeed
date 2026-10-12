// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "integration_test_helper.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace tokenspeed::test {

namespace {

struct ScheduledPrefill {
    std::int32_t extend_prefix_len{0};
    std::int32_t input_length{0};
};

// The prefill rows of the plan's model batch, by request id.
std::map<std::string, ScheduledPrefill> ScheduledPrefills(const ExecutionPlan& plan) {
    std::map<std::string, ScheduledPrefill> prefills;
    const ForwardBatch* batch = FindForwardBatch(plan);
    if (batch == nullptr) {
        return prefills;
    }
    for (std::size_t i = 0; i < batch->NumExtends(); ++i) {
        prefills[batch->request_ids[i]] = {batch->extend_prefix_lens[i], batch->input_lengths[i]};
    }
    return prefills;
}

std::vector<std::int32_t> Concat(std::vector<std::int32_t> head, const std::vector<std::int32_t>& tail) {
    head.insert(head.end(), tail.begin(), tail.end());
    return head;
}

RequestSpec Prompt(const std::string& id, std::vector<std::int32_t> tokens) {
    return RequestSpec{.request_id = id, .tokens = std::move(tokens)};
}

// Three pages (prefix granularity 2) shared by the prompts below; each prompt
// completes a fourth page of its own and ends one token into a fifth.
std::vector<std::int32_t> SharedPrefix() {
    return MakeTokens(6, 1);
}
std::vector<std::int32_t> PromptA() {
    return Concat(SharedPrefix(), {100, 101, 102});
}
std::vector<std::int32_t> PromptB() {
    return Concat(SharedPrefix(), {200, 201, 202});
}

}  // namespace

using InflightPrefixSharingTest = SchedulerTestSuite;

TEST_F(InflightPrefixSharingTest, SameRoundPromptWaitsForTheSharedPrefixAndHitsIt) {
    Submit({Prompt("a", PromptA()), Prompt("b", PromptB())});

    const auto first = ScheduledPrefills(PlanOnce());
    ASSERT_EQ(first.size(), 1u) << "b would compute the pages a computes in the same forward";
    EXPECT_EQ(first.at("a").extend_prefix_len, 0);
    EXPECT_EQ(first.at("a").input_length, 9);

    const auto second = ScheduledPrefills(PlanOnce());
    ASSERT_EQ(second.size(), 1u);
    EXPECT_EQ(second.at("b").extend_prefix_len, 6);
    EXPECT_EQ(second.at("b").input_length, 3);
}

TEST_F(InflightPrefixSharingTest, CompletedPrefillPublishesBeforeItsFirstDecode) {
    Submit(Prompt("a", PromptA()));
    ASSERT_EQ(ScheduledPrefills(PlanOnce()).size(), 1u);

    // Outside mixed mode a's first decode cannot run beside b's prefill, yet
    // b hits the prompt pages a computed.
    Submit(Prompt("b", PromptB()));
    const auto second = ScheduledPrefills(PlanOnce());
    ASSERT_EQ(second.size(), 1u);
    EXPECT_EQ(second.at("b").extend_prefix_len, 6);
}

TEST_F(InflightPrefixSharingTest, PromptsDivergingAfterTheCachedPrefixShareARound) {
    Submit(Prompt("seed", Concat(SharedPrefix(), {7})));
    PlanOnce();
    SendForwardDone("seed", {8});
    PlanOnce();
    SendFinish("seed");
    PlanOnce();

    Submit({Prompt("a", PromptA()), Prompt("b", PromptB())});
    const auto round = ScheduledPrefills(PlanOnce());
    ASSERT_EQ(round.size(), 2u) << "each computes only its own fourth page";
    EXPECT_EQ(round.at("a").extend_prefix_len, 6);
    EXPECT_EQ(round.at("b").extend_prefix_len, 6);
}

TEST_F(InflightPrefixSharingTest, IdenticalPromptsComputeThePromptOnce) {
    Submit({Prompt("a", MakeTokens(9, 1)), Prompt("b", MakeTokens(9, 1))});
    const auto first = ScheduledPrefills(PlanOnce());
    ASSERT_EQ(first.size(), 1u);
    EXPECT_TRUE(first.contains("a"));

    const auto second = ScheduledPrefills(PlanOnce());
    ASSERT_EQ(second.size(), 1u);
    EXPECT_EQ(second.at("b").extend_prefix_len, 8) << "only the final token is recomputed";
    EXPECT_EQ(second.at("b").input_length, 1);
}

TEST_F(InflightPrefixSharingTest, PromptAlreadyHittingThePageAWriterRecomputesDoesNotWait) {
    // A page-aligned prompt recomputes its cached last page -- a probe never
    // claims the page holding the final token. A prompt extending it already
    // hits that page, so it has nothing to wait for.
    const std::vector<std::int32_t> aligned = MakeTokens(8, 1);
    Submit(Prompt("seed", aligned));
    PlanOnce();
    SendForwardDone("seed", {9});
    PlanOnce();
    SendFinish("seed");
    PlanOnce();

    Submit({Prompt("a", aligned), Prompt("b", Concat(aligned, {50, 51, 52}))});
    const auto round = ScheduledPrefills(PlanOnce());
    ASSERT_EQ(round.size(), 2u);
    EXPECT_EQ(round.at("a").extend_prefix_len, 6);
    EXPECT_EQ(round.at("a").input_length, 2);
    EXPECT_EQ(round.at("b").extend_prefix_len, 8);
}

TEST_F(InflightPrefixSharingTest, LaterPromptsOfASharedPrefixHitTogether) {
    // C-Eval-shaped: many prompts share a 5-shot prefix. The first computes
    // it; the rest wait this round and hit together on the next.
    Submit({Prompt("a", PromptA()), Prompt("b", PromptB()), Prompt("c", Concat(SharedPrefix(), {300, 301, 302}))});
    ASSERT_EQ(ScheduledPrefills(PlanOnce()).size(), 1u);

    const auto second = ScheduledPrefills(PlanOnce());
    ASSERT_EQ(second.size(), 2u);
    EXPECT_EQ(second.at("b").extend_prefix_len, 6);
    EXPECT_EQ(second.at("c").extend_prefix_len, 6);
}

TEST_F(InflightPrefixSharingTest, WaitingPromptDoesNotHoldBackUnrelatedPrompts) {
    Submit({Prompt("a", PromptA()), Prompt("b", PromptB()), Prompt("c", MakeTokens(9, 500))});
    const auto first = ScheduledPrefills(PlanOnce());
    ASSERT_EQ(first.size(), 2u);
    EXPECT_TRUE(first.contains("a"));
    EXPECT_TRUE(first.contains("c"));
}

class InflightPrefixSharingWithoutPrefixCacheTest : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerTestSuite::MakeConfig();
        cfg.disable_prefix_cache = true;
        return cfg;
    }
};

TEST_F(InflightPrefixSharingWithoutPrefixCacheTest, PromptsSharingAPrefixShareARound) {
    Submit({Prompt("a", PromptA()), Prompt("b", PromptB())});
    EXPECT_EQ(ScheduledPrefills(PlanOnce()).size(), 2u) << "without prefix reuse there is nothing to wait for";
}

class InflightPrefixSharingStateCacheTest : public KimiFourGroupSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = KimiFourGroupSuite::MakeConfig();
        cfg.device_allocator.total_pages = 129;
        for (CacheGroupConfig& group : cfg.cache_groups) {
            group.total_pages = cfg.device_allocator.total_pages;
        }
        return cfg;
    }
};

class InflightPrefixSharingMixedTest : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerTestSuite::MakeConfig();
        cfg.enable_mixed_prefill_decode = true;
        return cfg;
    }
};

TEST_F(InflightPrefixSharingMixedTest, SameRoundPromptWaitsForTheSharedPrefixAndHitsIt) {
    Submit({Prompt("a", PromptA()), Prompt("b", PromptB())});

    const auto first = ScheduledPrefills(PlanOnce());
    ASSERT_EQ(first.size(), 1u);
    EXPECT_EQ(first.at("a").extend_prefix_len, 0);

    const auto second = ScheduledPrefills(PlanOnce());
    ASSERT_EQ(second.size(), 1u);
    EXPECT_EQ(second.at("b").extend_prefix_len, 6);
}

TEST_F(InflightPrefixSharingMixedTest, PrefillHitJoinsTheWritersFirstDecode) {
    Submit(Prompt("a", PromptA()));
    ASSERT_EQ(ScheduledPrefills(PlanOnce()).size(), 1u);

    Submit(Prompt("b", PromptB()));
    const ExecutionPlan plan = PlanOnce();
    const auto prefills = ScheduledPrefills(plan);
    ASSERT_EQ(prefills.size(), 1u);
    EXPECT_EQ(prefills.at("b").extend_prefix_len, 6);
    const ForwardBatch* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->decode_input_ids.size(), 1u) << "a's first decode rides the same mixed plan";
}

class InflightPrefixSharingChunkedTest : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerTestSuite::MakeConfig();
        // Four tokens leave a 9-token prompt incomplete; the writer holds the
        // line, so the shared-prefix waiter cannot sneak into this round.
        cfg.max_scheduled_tokens = 4;
        return cfg;
    }
};

TEST_F(InflightPrefixSharingChunkedTest, IncompleteChunkHoldsTheLineThenTheSharedPrefixHits) {
    Submit({Prompt("a", PromptA()), Prompt("b", PromptB())});

    const ExecutionPlan first = PlanOnce();
    const auto first_prefills = ScheduledPrefills(first);
    ASSERT_EQ(first_prefills.size(), 1u);
    EXPECT_TRUE(first_prefills.contains("a"));
    EXPECT_LT(first_prefills.at("a").input_length, 9);

    // Incomplete chunks hold the line. The finishing chunk becomes PrefillDone
    // in the same applyEvent, so HOL lifts and b joins that plan (1.1) and
    // hits the pages a's Admit already published. There is no b-only prefill.
    SendForwardDone("a");
    std::map<std::string, ScheduledPrefill> later;
    bool b_joined_finishing_chunk = false;
    for (int step = 0; step < 8; ++step) {
        later = ScheduledPrefills(PlanOnce());
        if (later.contains("b")) {
            ASSERT_TRUE(later.contains("a"));
            ASSERT_EQ(later.size(), 2u);
            EXPECT_EQ(later.at("b").extend_prefix_len, 6);
            b_joined_finishing_chunk = true;
            break;
        }
        ASSERT_EQ(later.size(), 1u);
        ASSERT_TRUE(later.contains("a"));
        SendForwardDone("a");
    }
    EXPECT_TRUE(b_joined_finishing_chunk);
}

TEST_F(InflightPrefixSharingStateCacheTest, OnlyAPromptSharingTheWritersCheckpointWaits) {
    // A state group resumes only where a checkpoint was published: a's
    // completed prompt leaves one at its last full page, 8 tokens in. b
    // diverges before it and could not resume there, so it does not wait;
    // c extends a's first 8 tokens and does.
    const std::vector<std::int32_t> a = PromptA();
    const std::vector<std::int32_t> c = Concat({a.begin(), a.begin() + 8}, {300, 301, 302});
    Submit({Prompt("a", a), Prompt("b", PromptB()), Prompt("c", c)});

    const auto first = ScheduledPrefills(PlanOnce());
    ASSERT_EQ(first.size(), 2u);
    EXPECT_TRUE(first.contains("a"));
    EXPECT_TRUE(first.contains("b"));

    const auto second = ScheduledPrefills(PlanOnce());
    ASSERT_EQ(second.size(), 1u);
    EXPECT_EQ(second.at("c").extend_prefix_len, 8);
    EXPECT_EQ(second.at("c").input_length, 3);
}

}  // namespace tokenspeed::test
