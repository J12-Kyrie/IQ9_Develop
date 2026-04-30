/// @file test_deduper_unit.cpp
/// @brief Group A: Pure C++ unit tests for Deduper logic (no GStreamer dependency)
///
/// Tests:
///   A1 - Whitelist (IsAllowed)
///   A2 - New target → interesting
///   A3 - Same target same position → not interesting
///   A4 - Target moved significantly → interesting
///   A5 - Target disappeared → interesting
///   A6 - Consecutive empty after disappear → not interesting
///   A7 - Reset clears state
///   A8 - IoU threshold boundary
///   A9 - Zero track_id → skipped
///   A10 - Multiple targets mixed scenario

#include "Deduper.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

static int g_pass = 0, g_fail = 0;

#define TEST_BEGIN(name) fprintf(stderr, "\n--- Test %s ---\n", name)
#define CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "  FAIL: %s\n", msg); \
            g_fail++; \
            return; \
        } \
    } while (0)
#define TEST_PASS(name) \
    do { \
        fprintf(stderr, "  PASS: %s\n", name); \
        g_pass++; \
    } while (0)

/* --- A1: Whitelist --- */
static void test_whitelist() {
    TEST_BEGIN("A1_whitelist");
    CHECK(deduper::Deduper::IsAllowed("person"),   "person allowed");
    CHECK(deduper::Deduper::IsAllowed("car"),       "car allowed");
    CHECK(deduper::Deduper::IsAllowed("bicycle"),   "bicycle allowed");
    CHECK(deduper::Deduper::IsAllowed("backpack"),  "backpack allowed");
    CHECK(deduper::Deduper::IsAllowed("umbrella"),  "umbrella allowed");
    CHECK(deduper::Deduper::IsAllowed("handbag"),   "handbag allowed");
    CHECK(!deduper::Deduper::IsAllowed("dog"),      "dog NOT allowed");
    CHECK(!deduper::Deduper::IsAllowed("cat"),      "cat NOT allowed");
    CHECK(!deduper::Deduper::IsAllowed("truck"),    "truck NOT allowed");
    CHECK(!deduper::Deduper::IsAllowed(""),         "empty NOT allowed");
    CHECK(!deduper::Deduper::IsAllowed("Person"),   "case-sensitive: Person NOT allowed");
    TEST_PASS("A1_whitelist");
}

/* --- A2: New target → interesting --- */
static void test_new_target() {
    TEST_BEGIN("A2_new_target");
    deduper::Deduper d;
    std::vector<deduper::TrackEntry> entries = {
        {"person", 1, 0.1, 0.2, 0.3, 0.4}
    };
    CHECK(d.IsInteresting(entries), "first frame with new target must be interesting");
    TEST_PASS("A2_new_target");
}

/* --- A3: Same target same position → not interesting --- */
static void test_no_change() {
    TEST_BEGIN("A3_no_change");
    deduper::Deduper d;
    std::vector<deduper::TrackEntry> entries = {
        {"person", 1, 0.1, 0.2, 0.3, 0.4}
    };
    d.IsInteresting(entries);  // frame 1: new → interesting
    CHECK(!d.IsInteresting(entries), "same position must NOT be interesting");
    TEST_PASS("A3_no_change");
}

/* --- A4: Target moved significantly → interesting --- */
static void test_target_moved() {
    TEST_BEGIN("A4_target_moved");
    deduper::DeduperConfig cfg;
    cfg.iou_threshold = 0.75f;
    deduper::Deduper d(cfg);

    std::vector<deduper::TrackEntry> e1 = {{"person", 1, 0.1, 0.1, 0.3, 0.3}};
    d.IsInteresting(e1);  // frame 1: new
    // 大幅移动: box 从 [0.1,0.1,0.3,0.3] 移到 [0.5,0.5,0.7,0.7], IoU ≈ 0 < 0.75
    std::vector<deduper::TrackEntry> e2 = {{"person", 1, 0.5, 0.5, 0.7, 0.7}};
    CHECK(d.IsInteresting(e2), "significant movement must be interesting");
    TEST_PASS("A4_target_moved");
}

/* --- A5: Target disappeared → interesting --- */
static void test_target_disappeared() {
    TEST_BEGIN("A5_target_disappeared");
    deduper::Deduper d;
    std::vector<deduper::TrackEntry> e1 = {{"person", 1, 0.1, 0.2, 0.3, 0.4}};
    d.IsInteresting(e1);

    std::vector<deduper::TrackEntry> empty = {};
    CHECK(d.IsInteresting(empty), "target disappearing must be interesting (Rule 3)");
    TEST_PASS("A5_target_disappeared");
}

/* --- A6: Consecutive empty after disappear → not interesting --- */
static void test_empty_after_disappear() {
    TEST_BEGIN("A6_empty_stable");
    deduper::Deduper d;
    std::vector<deduper::TrackEntry> e1 = {{"person", 1, 0.1, 0.2, 0.3, 0.4}};
    d.IsInteresting(e1);            // new → interesting, state = {person#1}

    std::vector<deduper::TrackEntry> empty = {};
    d.IsInteresting(empty);         // disappeared → interesting, state = {}
    CHECK(!d.IsInteresting(empty),  "second empty frame must NOT be interesting");
    TEST_PASS("A6_empty_stable");
}

/* --- A7: Reset clears state --- */
static void test_reset() {
    TEST_BEGIN("A7_reset");
    deduper::Deduper d;
    std::vector<deduper::TrackEntry> e1 = {{"person", 1, 0.1, 0.2, 0.3, 0.4}};
    d.IsInteresting(e1);
    CHECK(!d.IsInteresting(e1), "same frame not interesting before reset");

    d.Reset();
    CHECK(d.IsInteresting(e1), "after Reset, same target must be interesting again");
    TEST_PASS("A7_reset");
}

/* --- A8: IoU threshold boundary --- */
static void test_iou_boundary() {
    TEST_BEGIN("A8_iou_boundary");
    deduper::DeduperConfig cfg;
    cfg.iou_threshold = 0.5f;
    deduper::Deduper d(cfg);

    // Box1: [0, 0, 1, 1]  area = 1.0
    std::vector<deduper::TrackEntry> e1 = {{"car", 1, 0.0, 0.0, 1.0, 1.0}};
    d.IsInteresting(e1);

    // Box2: [0.6, 0, 1.2, 1]  area = 0.6
    // overlap: x1=0.6, y1=0, x2=1.0, y2=1.0 → w=0.4, h=1.0 → 0.4
    // union: 1.0 + 0.6 - 0.4 = 1.2
    // IoU = 0.4 / 1.2 = 0.333 < 0.5 → interesting
    std::vector<deduper::TrackEntry> e2 = {{"car", 1, 0.6, 0.0, 1.2, 1.0}};
    CHECK(d.IsInteresting(e2), "IoU=0.33 < 0.5 threshold → must be interesting");

    // Box3: tiny shift from Box2 → IoU ≈ 1.0 > 0.5 → not interesting
    std::vector<deduper::TrackEntry> e3 = {{"car", 1, 0.601, 0.0, 1.201, 1.0}};
    CHECK(!d.IsInteresting(e3), "tiny shift (IoU≈1.0 > threshold) must NOT be interesting");
    TEST_PASS("A8_iou_boundary");
}

/* --- A9: Zero track_id → skipped --- */
static void test_zero_track_id() {
    TEST_BEGIN("A9_zero_track_id");
    deduper::Deduper d;
    // track_id == 0 → skipped by IsInteresting
    std::vector<deduper::TrackEntry> e1 = {{"person", 0, 0.1, 0.2, 0.3, 0.4}};
    CHECK(!d.IsInteresting(e1), "track_id=0 should be skipped → empty map → not interesting");
    // Even second call should not be interesting (state never updated)
    CHECK(!d.IsInteresting(e1), "still not interesting with track_id=0");
    TEST_PASS("A9_zero_track_id");
}

/* --- A10: Multiple targets mixed scenario --- */
static void test_multi_target() {
    TEST_BEGIN("A10_multi_target");
    deduper::Deduper d;

    // Frame 1: person#1 + car#2 → new → interesting
    std::vector<deduper::TrackEntry> f1 = {
        {"person", 1, 0.1, 0.1, 0.3, 0.3},
        {"car",    2, 0.5, 0.5, 0.8, 0.8}
    };
    CHECK(d.IsInteresting(f1), "frame1: two new targets → interesting");

    // Frame 2: same positions → not interesting
    CHECK(!d.IsInteresting(f1), "frame2: no change → not interesting");

    // Frame 3: person#1 moved, car#2 same → interesting (Rule 2)
    std::vector<deduper::TrackEntry> f3 = {
        {"person", 1, 0.4, 0.4, 0.6, 0.6},  // moved significantly
        {"car",    2, 0.5, 0.5, 0.8, 0.8}    // same
    };
    CHECK(d.IsInteresting(f3), "frame3: person moved → interesting");

    // Frame 4: car#2 disappeared → interesting (Rule 3)
    std::vector<deduper::TrackEntry> f4 = {
        {"person", 1, 0.4, 0.4, 0.6, 0.6}   // same position
    };
    CHECK(d.IsInteresting(f4), "frame4: car disappeared → interesting");

    // Frame 5: person#1 same → not interesting
    CHECK(!d.IsInteresting(f4), "frame5: no change → not interesting");

    // Frame 6: new bicycle#3 appears → interesting (Rule 1)
    std::vector<deduper::TrackEntry> f6 = {
        {"person",  1, 0.4, 0.4, 0.6, 0.6},
        {"bicycle", 3, 0.7, 0.1, 0.9, 0.3}
    };
    CHECK(d.IsInteresting(f6), "frame6: bicycle appears → interesting");

    TEST_PASS("A10_multi_target");
}

/* --- A11: IsInteresting() benchmark --- */
static void test_benchmark() {
    TEST_BEGIN("A11_benchmark");

    deduper::DeduperConfig cfg;
    cfg.iou_threshold = 0.75f;

    std::vector<deduper::TrackEntry> entries_3 = {
        {"person",  1, 0.10, 0.10, 0.30, 0.40},
        {"car",     2, 0.50, 0.30, 0.80, 0.70},
        {"bicycle", 3, 0.20, 0.60, 0.40, 0.90},
    };
    std::vector<deduper::TrackEntry> entries_3_moved = {
        {"person",  1, 0.15, 0.15, 0.35, 0.45},
        {"car",     2, 0.55, 0.35, 0.85, 0.75},
        {"bicycle", 3, 0.25, 0.65, 0.45, 0.95},
    };

    constexpr int kWarmup = 200;
    constexpr int kIters  = 10000;

    {
        deduper::Deduper d(cfg);
        for (int i = 0; i < kWarmup; i++) {
            d.Reset();
            d.IsInteresting(entries_3);
        }
    }

    double min_us = 1e9, max_us = 0;
    deduper::Deduper d(cfg);
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kIters; i++) {
        d.Reset();
        auto s = std::chrono::steady_clock::now();
        d.IsInteresting(entries_3);
        d.IsInteresting(entries_3);        // not interesting (same pos)
        d.IsInteresting(entries_3_moved);   // interesting (moved)
        auto e = std::chrono::steady_clock::now();
        double us = std::chrono::duration<double, std::micro>(e - s).count() / 3.0;
        if (us < min_us) min_us = us;
        if (us > max_us) max_us = us;
    }
    auto t1 = std::chrono::steady_clock::now();
    double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double avg_us = total_us / (kIters * 3.0);

    fprintf(stderr, "  IsInteresting() benchmark: %d calls (3 entries each)\n", kIters * 3);
    fprintf(stderr, "  avg=%.2f us  min=%.2f us  max=%.2f us\n", avg_us, min_us, max_us);

    CHECK(avg_us < 1000.0, "avg should be < 1ms per call");
    TEST_PASS("A11_benchmark");
}

int main() {
    fprintf(stderr, "========================================\n");
    fprintf(stderr, "=== Deduper Unit Tests (Group A) ===\n");
    fprintf(stderr, "========================================\n");

    test_whitelist();
    test_new_target();
    test_no_change();
    test_target_moved();
    test_target_disappeared();
    test_empty_after_disappear();
    test_reset();
    test_iou_boundary();
    test_zero_track_id();
    test_multi_target();
    test_benchmark();

    fprintf(stderr, "\n========================================\n");
    fprintf(stderr, "=== SUMMARY: %d PASSED, %d FAILED ===\n", g_pass, g_fail);
    fprintf(stderr, "========================================\n");
    return g_fail > 0 ? 1 : 0;
}
