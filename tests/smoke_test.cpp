#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

namespace {
constexpr int placeholder_success_code = 0;
}

TEST_CASE("quality tooling smoke test") { CHECK(placeholder_success_code == 0); }
