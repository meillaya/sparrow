// ForgeRT C++ ABI — gtest round-trip suite (spec §10).
// Covers every public header introduced in Task 10:
//   - TensorDescriptor (round-trip + non-contiguous detection)
//   - ArtifactDescriptor (binary + argument + guard counts)
//   - Buffer (move-only + malloc/free ownership)
//   - Event (wait/timeout/complete semantics, Phase 1 CPU stub)
#include "forge_rt/forge_rt.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

using forge_rt::ArtifactDescriptor;
using forge_rt::ArgumentDescriptor;
using forge_rt::Buffer;
using forge_rt::BufferSpec;
using forge_rt::DataType;
using forge_rt::Event;
using forge_rt::MemorySpace;
using forge_rt::ShapeGuard;
using forge_rt::ShapeGuardKind;
using forge_rt::TargetKind;
using forge_rt::TensorDescriptor;
using forge_rt::byte_size;
using forge_rt::is_contiguous;
using forge_rt::numel;

// ---------------------------------------------------------------------------
// 1. TensorDescriptor round-trip — [2, 512, 4096] f32 contiguous on Host.
// ---------------------------------------------------------------------------
TEST(TensorDescriptor_roundtrip, BuildsContiguousHostF32) {
    TensorDescriptor td;
    td.dtype = DataType::Float32;
    td.shape = {2, 512, 4096};
    td.strides = {512 * 4096, 4096, 1};  // row-major contiguous
    td.memory_space = MemorySpace::Host;
    td.byte_offset = 0;

    constexpr std::size_t kExpectedNumel = 2ULL * 512ULL * 4096ULL;
    constexpr std::size_t kExpectedBytes = kExpectedNumel * 4ULL;  // Float32

    EXPECT_EQ(numel(td), kExpectedNumel);
    EXPECT_EQ(byte_size(td), kExpectedBytes);
    EXPECT_TRUE(is_contiguous(td));
}

// ---------------------------------------------------------------------------
// 2. TensorDescriptor non-contiguous — transposed strides break contiguity.
// ---------------------------------------------------------------------------
TEST(TensorDescriptor_noncontiguous, TransposedStridesBreakContiguity) {
    TensorDescriptor td;
    td.dtype = DataType::Float32;
    td.shape = {4, 8};
    td.memory_space = MemorySpace::Host;

    // Row-major contiguous → stride = (8, 1).
    td.strides = {8, 1};
    EXPECT_TRUE(is_contiguous(td));

    // Transposed [8, 4] view of the same buffer: stride = (1, 8).
    td.shape = {8, 4};
    td.strides = {1, 8};
    EXPECT_FALSE(is_contiguous(td));

    // Wrong rank strides also fail.
    TensorDescriptor bad = td;
    bad.strides = {1};  // only 1 stride for a rank-2 tensor
    EXPECT_FALSE(is_contiguous(bad));
}

// ---------------------------------------------------------------------------
// 3. ArtifactDescriptor size — binary blob, 3 args, 2 shape guards.
// ---------------------------------------------------------------------------
TEST(ArtifactDescriptor_size, BuildsWithBinaryArgsGuards) {
    ArtifactDescriptor ad;
    ad.target = TargetKind::CPU;
    ad.entry_point = "fused_gemm_bias_gelu";

    // 16-byte binary blob (opaque kernel artifact for Phase 1 stub).
    ad.binary.resize(16);
    for (std::size_t i = 0; i < ad.binary.size(); ++i) {
        ad.binary[i] = static_cast<std::byte>(i);
    }

    ad.arguments = {
        ArgumentDescriptor{"a", DataType::Float32, /*is_input=*/true, {2, 512, 4096}},
        ArgumentDescriptor{"b", DataType::Float32, /*is_input=*/true, {2, 4096, 11008}},
        ArgumentDescriptor{"y", DataType::Float32, /*is_input=*/false, {2, 512, 11008}},
    };

    ad.guards = {
        ShapeGuard{{0, 1}, ShapeGuardKind::Equal, 2},          // batch dim == 2
        ShapeGuard{{2}, ShapeGuardKind::Divisible, 128},       // inner dim divisible by 128
    };

    ad.workspace_bytes = 1u << 20;  // 1 MiB scratch

    EXPECT_EQ(ad.binary.size(), 16u);
    EXPECT_EQ(ad.arguments.size(), 3u);
    EXPECT_EQ(ad.guards.size(), 2u);
    EXPECT_EQ(ad.workspace_bytes, 1u << 20);

    // Argument inspection round-trip.
    EXPECT_EQ(ad.arguments[0].name, "a");
    EXPECT_EQ(ad.arguments[0].shape.size(), 3u);
    EXPECT_TRUE(ad.arguments[0].is_input);
    EXPECT_FALSE(ad.arguments[2].is_input);

    // Guard inspection.
    EXPECT_EQ(ad.guards[0].kind, ShapeGuardKind::Equal);
    EXPECT_EQ(ad.guards[1].kind, ShapeGuardKind::Divisible);
    EXPECT_EQ(ad.guards[1].value, 128);
}

// ---------------------------------------------------------------------------
// 4. Buffer move-only — owns=true malloc'd memory; move transfers ownership.
// ---------------------------------------------------------------------------
TEST(Buffer_move_only, MoveTransfersOwnershipAndReleasesOnce) {
    constexpr std::size_t kBytes = 64;

    void* raw = std::malloc(kBytes);
    ASSERT_NE(raw, nullptr);
    std::memset(raw, 0xAB, kBytes);

    BufferSpec spec;
    spec.descriptor.dtype = DataType::Float32;
    spec.descriptor.shape = {16};
    spec.descriptor.memory_space = MemorySpace::Host;
    spec.byte_size = kBytes;
    spec.alignment = alignof(std::max_align_t);

    Buffer src(raw, spec, /*owns=*/true);
    EXPECT_EQ(src.data(), raw);
    EXPECT_TRUE(src.owns());

    Buffer dst{std::move(src)};
    EXPECT_EQ(dst.data(), raw);
    EXPECT_TRUE(dst.owns());
    EXPECT_EQ(dst.spec().byte_size, kBytes);

    // Source is now empty.
    EXPECT_EQ(src.data(), nullptr);
    EXPECT_FALSE(src.owns());

    // Move-assign onto a fresh default buffer.
    Buffer third;
    third = std::move(dst);
    EXPECT_EQ(third.data(), raw);
    EXPECT_TRUE(third.owns());
    EXPECT_EQ(dst.data(), nullptr);

    // `third` owns the malloc'd memory; its destructor will std::free(raw)
    // exactly once. No leaks, no double-free. (Verified by the absence of
    // sanitizer/ASan reports — the test itself asserts structural move
    // semantics; leak detection is a CI concern.)
}

// ---------------------------------------------------------------------------
// 5. Event wait — not completed + zero timeout → Error; mark_completed → ok.
// ---------------------------------------------------------------------------
TEST(Event_wait_completes, ZeroTimeoutBeforeCompleteReturnsError) {
    Event ev(42);
    EXPECT_FALSE(ev.completed());

    auto r = ev.wait(std::chrono::milliseconds{0});
    ASSERT_FALSE(r.ok());
    EXPECT_EQ(r.error().code, -1);

    // Mark complete, then a 0-timeout wait must succeed.
    ev.mark_completed();
    EXPECT_TRUE(ev.completed());
    auto r2 = ev.wait(std::chrono::milliseconds{0});
    EXPECT_TRUE(r2.ok());
}

TEST(EventCopy, CopiesShareCompletionState) {
    Event original(3);
    Event copy = original;
    EXPECT_FALSE(original.completed());
    EXPECT_FALSE(copy.completed());

    copy.mark_completed();
    EXPECT_TRUE(original.completed());
    EXPECT_TRUE(copy.completed());
}

TEST(Event_wait_completes, PositiveTimeoutPollsUntilComplete) {
    Event ev(7);
    EXPECT_FALSE(ev.completed());

    // Mark complete in 25ms; the wait(100ms) must return ok.
    std::thread([&ev]() {
        std::this_thread::sleep_for(std::chrono::milliseconds{25});
        ev.mark_completed();
    }).detach();

    auto r = ev.wait(std::chrono::milliseconds{100});
    EXPECT_TRUE(r.ok());
    EXPECT_TRUE(ev.completed());
}
