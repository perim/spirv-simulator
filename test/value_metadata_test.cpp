#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "spirv_simulator.hpp"
#include "testing_common.hpp"

using namespace testing;

// verify that OpCompositeConstruct produces the compact vector
// representation and the generic aggregate scalar-leaf representation.
class CompositeMetadataTests : public SPIRVSimulatorMockBase, public Test
{
  protected:
    void SetValueWithBaseImplementation(uint32_t                       result_id,
                                        const ::SPIRVSimulator::Value& value,
                                        bool                           clear_meta)
    {
        ::SPIRVSimulator::SPIRVSimulator::SetValue(result_id, value, clear_meta);
    }
};


TEST_F(CompositeMetadataTests, VectorConstructPreservesComponentRanges)
{
    TestParameters parameters =
        TestParametersBuilder()
            .set_opcode(spv::Op::OpCompositeConstruct)
            .set_operand_at(
                0,
                std::make_shared<::SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 1, 2 }),
                CommonTypes::uvec2)
            .set_operands_range(1, CommonTypes::u64, std::initializer_list<uint64_t>{ 1, 2 })
            .build();

    std::vector<uint32_t> words = prepare_submission(parameters);
    ::SPIRVSimulator::Instruction instruction{ .opcode     = parameters.opcode,
                                               .word_count = static_cast<uint16_t>(words.size()),
                                               .words      = words };

    const uint32_t result_id = words[2];
    const uint32_t first_component_id = words[3];
    const uint32_t second_component_id = words[4];
    value_meta_[first_component_id].value_range =
        ::SPIRVSimulator::DenseValueRangeInfo{ true, true, true, 0, 7, 1 };
    value_meta_[second_component_id].value_range =
        ::SPIRVSimulator::DenseValueRangeInfo{ true, true, true, 8, 15, 1 };

    EXPECT_CALL(*this, SetValue(result_id, _, true))
        .WillOnce([this](uint32_t id, const ::SPIRVSimulator::Value& value, bool clear_meta) {
            SetValueWithBaseImplementation(id, value, clear_meta);
        });

    EXPECT_TRUE(ExecuteInstruction(instruction));

    const auto& ranges = value_meta_[result_id].component_ranges;
    ASSERT_EQ(ranges.size(), 2u);
    EXPECT_EQ(ranges[0].min, 0u);
    EXPECT_EQ(ranges[0].max, 7u);
    EXPECT_EQ(ranges[1].min, 8u);
    EXPECT_EQ(ranges[1].max, 15u);
}

TEST_F(CompositeMetadataTests, StructConstructPreservesSubobjectScalarRanges)
{
    auto offset = std::make_shared<::SPIRVSimulator::VectorV>(std::initializer_list<uint64_t>{ 3, 4 });
    TestParameters parameters =
        TestParametersBuilder()
            .set_opcode(spv::Op::OpCompositeConstruct)
            .set_operand_at(
                0,
                std::make_shared<::SPIRVSimulator::AggregateV>(
                    std::initializer_list<::SPIRVSimulator::Value>{ uint64_t{ 2 }, offset }),
                ::SPIRVSimulator::Type::Struct(0))
            .set_operand_at(1, uint64_t{ 2 }, CommonTypes::u64)
            .set_operand_at(2, offset, CommonTypes::uvec2)
            .build();

    std::vector<uint32_t> words = prepare_submission(parameters);
    ::SPIRVSimulator::Instruction instruction{ .opcode     = parameters.opcode,
                                               .word_count = static_cast<uint16_t>(words.size()),
                                               .words      = words };

    const uint32_t result_id = words[2];
    const uint32_t scalar_id = words[3];
    const uint32_t vector_id = words[4];
    value_meta_[scalar_id].value_range =
        ::SPIRVSimulator::DenseValueRangeInfo{ true, true, true, 0, 7, 1 };
    value_meta_[vector_id].component_ranges = {
        ::SPIRVSimulator::DenseValueRangeInfo{ true, true, true, 0, 15, 1 },
        ::SPIRVSimulator::DenseValueRangeInfo{ true, true, true, 0, 31, 1 },
    };

    EXPECT_CALL(*this, SetValue(result_id, _, true))
        .WillOnce([this](uint32_t id, const ::SPIRVSimulator::Value& value, bool clear_meta) {
            SetValueWithBaseImplementation(id, value, clear_meta);
        });

    EXPECT_TRUE(ExecuteInstruction(instruction));

    const auto& ranges = value_meta_[result_id].subobject_scalar_ranges;
    ASSERT_EQ(ranges.size(), 3u);
    EXPECT_EQ(ranges[0].path, std::vector<uint32_t>{ 0 });
    EXPECT_EQ(ranges[0].value_range.max, 7u);
    EXPECT_EQ(ranges[1].path, (std::vector<uint32_t>{ 1, 0 }));
    EXPECT_EQ(ranges[1].value_range.max, 15u);
    EXPECT_EQ(ranges[2].path, (std::vector<uint32_t>{ 1, 1 }));
    EXPECT_EQ(ranges[2].value_range.max, 31u);
}

namespace
{

// Shared semantic cases used by UT-CM-03 and UT-CM-04. Each case describes
// the same subtree independently of whether it is reached by extract or load.
enum class SubtreeProjectionShape
{
    StructMemberVector,
    StructMemberArray,
    NestedStructSubtree,
};

struct ExpectedValueMetadataSpec
{
    ::SPIRVSimulator::DenseValueRangeInfo value_range;
    std::vector<::SPIRVSimulator::DenseValueRangeInfo> component_ranges;
    std::vector<::SPIRVSimulator::SubobjectScalarRangeEntry> subobject_scalar_ranges;
};

struct SubtreeProjectionTestCase
{
    const char* name;
    SubtreeProjectionShape shape;
    std::vector<uint32_t> selected_path;
    std::vector<::SPIRVSimulator::SubobjectScalarRangeEntry> source_subobject_ranges;
    ExpectedValueMetadataSpec expected_extracted_metadata;
    std::vector<uint32_t> followup_scalar_path;
    ::SPIRVSimulator::DenseValueRangeInfo expected_scalar_range;
    uint64_t expected_scalar_value;
};

::SPIRVSimulator::DenseValueRangeInfo MakeProjectionRange(bool thread_dependent,
                                                          bool dense_range,
                                                          uint64_t min,
                                                          uint64_t max,
                                                          uint64_t stride)
{
    return ::SPIRVSimulator::DenseValueRangeInfo{
        true,
        thread_dependent,
        dense_range,
        min,
        max,
        stride,
    };
}

SubtreeProjectionTestCase MakeStructMemberVectorCase()
{
    const auto range_a = MakeProjectionRange(true, true, 10, 19, 1);
    const auto range_b = MakeProjectionRange(true, true, 30, 38, 2);
    const auto sibling_range = MakeProjectionRange(false, true, 100, 100, 1);

    return SubtreeProjectionTestCase{
        "StructMemberVector",
        SubtreeProjectionShape::StructMemberVector,
        { 0 },
        {
            { { 0, 0 }, range_a },
            { { 0, 1 }, range_b },
            { { 1 }, sibling_range },
        },
        {
            {},
            { range_a, range_b },
            {},
        },
        { 1 },
        range_b,
        22,
    };
}

SubtreeProjectionTestCase MakeStructMemberArrayCase()
{
    const auto range_a = MakeProjectionRange(true, true, 40, 49, 1);
    const auto range_b = MakeProjectionRange(true, false, 60, 72, 4);
    const auto sibling_range = MakeProjectionRange(false, true, 110, 110, 1);

    return SubtreeProjectionTestCase{
        "StructMemberArray",
        SubtreeProjectionShape::StructMemberArray,
        { 1 },
        {
            { { 1, 0 }, range_a },
            { { 1, 1 }, range_b },
            { { 0 }, sibling_range },
        },
        {
            {},
            {},
            {
                { { 0 }, range_a },
                { { 1 }, range_b },
            },
        },
        { 0 },
        range_a,
        31,
    };
}

SubtreeProjectionTestCase MakeNestedStructSubtreeCase()
{
    const auto range_a = MakeProjectionRange(true, true, 80, 95, 1);
    const auto range_b = MakeProjectionRange(false, false, 120, 132, 4);
    const auto sibling_range = MakeProjectionRange(true, true, 200, 209, 1);

    return SubtreeProjectionTestCase{
        "NestedStructSubtree",
        SubtreeProjectionShape::NestedStructSubtree,
        { 1 },
        {
            { { 1, 2, 0 }, range_a },
            { { 1, 2, 1 }, range_b },
            { { 3, 0 }, sibling_range },
        },
        {
            {},
            {},
            {
                { { 2, 0 }, range_a },
                { { 2, 1 }, range_b },
            },
        },
        { 2, 1 },
        range_b,
        42,
    };
}

const SubtreeProjectionTestCase kStructMemberVectorCase = MakeStructMemberVectorCase();
const SubtreeProjectionTestCase kStructMemberArrayCase = MakeStructMemberArrayCase();
const SubtreeProjectionTestCase kNestedStructSubtreeCase = MakeNestedStructSubtreeCase();

struct SubtreeProjectionValueSetup
{
    ::SPIRVSimulator::Value source_value;
    ::SPIRVSimulator::Value extracted_value;
    ::SPIRVSimulator::Type extracted_type;
    uint32_t extracted_type_id;
};

SubtreeProjectionValueSetup MakeSubtreeProjectionValueSetup(SubtreeProjectionShape shape)
{
    switch (shape)
    {
        case SubtreeProjectionShape::StructMemberVector:
        {
            auto selected = std::make_shared<::SPIRVSimulator::VectorV>(
                std::initializer_list<uint64_t>{ 11, 22 });
            auto source = std::make_shared<::SPIRVSimulator::AggregateV>(
                std::initializer_list<::SPIRVSimulator::Value>{ selected, uint64_t{ 99 } });
            return SubtreeProjectionValueSetup{
                source,
                selected,
                ::SPIRVSimulator::Type::Vector(CommonTypes::u64, 2),
                CommonTypes::uvec2,
            };
        }
        case SubtreeProjectionShape::StructMemberArray:
        {
            auto selected = std::make_shared<::SPIRVSimulator::AggregateV>(
                std::initializer_list<::SPIRVSimulator::Value>{ uint64_t{ 31 }, uint64_t{ 32 } });
            auto source = std::make_shared<::SPIRVSimulator::AggregateV>(
                std::initializer_list<::SPIRVSimulator::Value>{ uint64_t{ 98 }, selected });
            return SubtreeProjectionValueSetup{
                source,
                selected,
                ::SPIRVSimulator::Type::Array(CommonTypes::u64, 2),
                7010,
            };
        }
        case SubtreeProjectionShape::NestedStructSubtree:
        {
            auto nested_vector = std::make_shared<::SPIRVSimulator::VectorV>(
                std::initializer_list<uint64_t>{ 41, 42 });
            auto selected = std::make_shared<::SPIRVSimulator::AggregateV>(
                std::initializer_list<::SPIRVSimulator::Value>{
                    uint64_t{ 5 },
                    uint64_t{ 6 },
                    nested_vector,
                });
            auto sibling = std::make_shared<::SPIRVSimulator::AggregateV>(
                std::initializer_list<::SPIRVSimulator::Value>{ uint64_t{ 97 } });
            auto source = std::make_shared<::SPIRVSimulator::AggregateV>(
                std::initializer_list<::SPIRVSimulator::Value>{
                    uint64_t{ 4 },
                    selected,
                    uint64_t{ 7 },
                    sibling,
                });
            return SubtreeProjectionValueSetup{
                source,
                selected,
                ::SPIRVSimulator::Type::Struct(7011),
                7011,
            };
        }
    }

    return {};
}

void ExpectProjectionRange(const ::SPIRVSimulator::DenseValueRangeInfo& actual,
                           const ::SPIRVSimulator::DenseValueRangeInfo& expected)
{
    EXPECT_EQ(actual.valid, expected.valid);
    EXPECT_EQ(actual.thread_dependent, expected.thread_dependent);
    EXPECT_EQ(actual.dense_range, expected.dense_range);
    EXPECT_EQ(actual.min, expected.min);
    EXPECT_EQ(actual.max, expected.max);
    EXPECT_EQ(actual.stride, expected.stride);
}

void ExpectProjectionMetadata(const ::SPIRVSimulator::ValueMetadata& actual,
                              const ExpectedValueMetadataSpec& expected)
{
    ExpectProjectionRange(actual.value_range, expected.value_range);

    ASSERT_EQ(actual.component_ranges.size(), expected.component_ranges.size());
    for (size_t i = 0; i < expected.component_ranges.size(); ++i)
    {
        ExpectProjectionRange(actual.component_ranges[i], expected.component_ranges[i]);
    }

    ASSERT_EQ(actual.subobject_scalar_ranges.size(), expected.subobject_scalar_ranges.size());
    for (size_t i = 0; i < expected.subobject_scalar_ranges.size(); ++i)
    {
        EXPECT_EQ(actual.subobject_scalar_ranges[i].path,
                  expected.subobject_scalar_ranges[i].path);
        ExpectProjectionRange(actual.subobject_scalar_ranges[i].value_range,
                              expected.subobject_scalar_ranges[i].value_range);
    }

    EXPECT_FALSE(actual.address_range_valid);
}

void SeedStaleProjectionMetadata(::SPIRVSimulator::ValueMetadata& metadata)
{
    const auto stale_range = MakeProjectionRange(true, true, 900, 999, 1);
    metadata.value_range = stale_range;
    metadata.component_ranges = { stale_range };
    metadata.subobject_scalar_ranges = { { { 9 }, stale_range } };
    metadata.address_range_valid = true;
    metadata.address_range_min = 900;
    metadata.address_range_max = 999;
}

void ExpectProjectedConcreteValue(SubtreeProjectionShape shape,
                                  const ::SPIRVSimulator::Value& value)
{
    switch (shape)
    {
        case SubtreeProjectionShape::StructMemberVector:
        {
            ASSERT_TRUE(std::holds_alternative<std::shared_ptr<::SPIRVSimulator::VectorV>>(value));
            const auto& vector = *std::get<std::shared_ptr<::SPIRVSimulator::VectorV>>(value);
            ASSERT_EQ(vector.elems.size(), 2u);
            EXPECT_EQ(std::get<uint64_t>(vector.elems[0]), 11u);
            EXPECT_EQ(std::get<uint64_t>(vector.elems[1]), 22u);
            break;
        }
        case SubtreeProjectionShape::StructMemberArray:
        {
            ASSERT_TRUE(std::holds_alternative<std::shared_ptr<::SPIRVSimulator::AggregateV>>(value));
            const auto& array = *std::get<std::shared_ptr<::SPIRVSimulator::AggregateV>>(value);
            ASSERT_EQ(array.elems.size(), 2u);
            EXPECT_EQ(std::get<uint64_t>(array.elems[0]), 31u);
            EXPECT_EQ(std::get<uint64_t>(array.elems[1]), 32u);
            break;
        }
        case SubtreeProjectionShape::NestedStructSubtree:
        {
            ASSERT_TRUE(std::holds_alternative<std::shared_ptr<::SPIRVSimulator::AggregateV>>(value));
            const auto& aggregate = *std::get<std::shared_ptr<::SPIRVSimulator::AggregateV>>(value);
            ASSERT_EQ(aggregate.elems.size(), 3u);
            ASSERT_TRUE(std::holds_alternative<std::shared_ptr<::SPIRVSimulator::VectorV>>(
                aggregate.elems[2]));
            const auto& vector = *std::get<std::shared_ptr<::SPIRVSimulator::VectorV>>(
                aggregate.elems[2]);
            ASSERT_EQ(vector.elems.size(), 2u);
            EXPECT_EQ(std::get<uint64_t>(vector.elems[0]), 41u);
            EXPECT_EQ(std::get<uint64_t>(vector.elems[1]), 42u);
            break;
        }
    }
}

} // namespace

// execute aggregate-valued OpCompositeExtract followed by a scalar
// extract, proving that descendant metadata is rebased and remains consumable.
class CompositeExtractSubtreeMetadataTests : public SPIRVSimulatorMockBase,
                                             public TestWithParam<SubtreeProjectionTestCase>
{
  protected:
    static constexpr uint32_t kSourceId = 7000;
    static constexpr uint32_t kExtractedResultId = 7001;
    static constexpr uint32_t kScalarResultId = 7002;
    static constexpr uint32_t kMaximumTestId = 7002;

    void SetUp() override
    {
        // The mock-only default constructor does not initialize optional
        // runtime services used by production simulator construction.
        memory_flag_tracker_ = nullptr;
        verbose_ = false;

        if (values_.size() <= kMaximumTestId)
        {
            values_.resize(kMaximumTestId + 1);
        }
        if (value_meta_.size() <= kMaximumTestId)
        {
            value_meta_.resize(kMaximumTestId + 1);
        }

        const SubtreeProjectionValueSetup value_setup =
            MakeSubtreeProjectionValueSetup(GetParam().shape);
        values_[kSourceId] = value_setup.source_value;
        extracted_result_type_ = value_setup.extracted_type;
        extracted_result_type_id_ = value_setup.extracted_type_id;

        EXPECT_CALL(*this, GetValue(_))
            .WillRepeatedly([this](uint32_t id) -> ::SPIRVSimulator::Value& {
                return values_[id];
            });
        EXPECT_CALL(*this, SetValue(_, _, true))
            .Times(2)
            .WillRepeatedly([this](uint32_t id,
                                   const ::SPIRVSimulator::Value& value,
                                   bool clear_meta) {
                ::SPIRVSimulator::SPIRVSimulator::SetValue(id, value, clear_meta);
            });
        EXPECT_CALL(*this, GetTypeByResultId(kExtractedResultId))
            .WillRepeatedly(ReturnRef(extracted_result_type_));
        EXPECT_CALL(*this, GetTypeByResultId(kScalarResultId))
            .WillRepeatedly(ReturnRef(types_[CommonTypes::u64]));
        EXPECT_CALL(*this, TransferFlags(A<uint32_t>(), A<uint32_t>()))
            .Times(2);
    }

    bool ExecuteCompositeExtract(uint32_t result_type_id,
                                 uint32_t result_id,
                                 uint32_t composite_id,
                                 const std::vector<uint32_t>& indexes)
    {
        std::vector<uint32_t> words{
            static_cast<uint32_t>(spv::Op::OpCompositeExtract),
            result_type_id,
            result_id,
            composite_id,
        };
        words.insert(words.end(), indexes.begin(), indexes.end());

        const ::SPIRVSimulator::Instruction instruction{
            .opcode = spv::Op::OpCompositeExtract,
            .word_count = static_cast<uint16_t>(words.size()),
            .words = words,
        };
        return ExecuteInstruction(instruction);
    }

    ::SPIRVSimulator::Type extracted_result_type_;
    uint32_t extracted_result_type_id_ = 0;
};

TEST_P(CompositeExtractSubtreeMetadataTests, RebasesDescendantMetadata)
{
    value_meta_[kSourceId].subobject_scalar_ranges = GetParam().source_subobject_ranges;
    SeedStaleProjectionMetadata(value_meta_[kExtractedResultId]);
    SeedStaleProjectionMetadata(value_meta_[kScalarResultId]);

    ASSERT_TRUE(ExecuteCompositeExtract(extracted_result_type_id_,
                                        kExtractedResultId,
                                        kSourceId,
                                        GetParam().selected_path));

    ExpectProjectedConcreteValue(GetParam().shape, values_[kExtractedResultId]);
    ExpectProjectionMetadata(value_meta_[kExtractedResultId],
                             GetParam().expected_extracted_metadata);

    ASSERT_TRUE(ExecuteCompositeExtract(CommonTypes::u64,
                                        kScalarResultId,
                                        kExtractedResultId,
                                        GetParam().followup_scalar_path));

    ASSERT_TRUE(std::holds_alternative<uint64_t>(values_[kScalarResultId]));
    EXPECT_EQ(std::get<uint64_t>(values_[kScalarResultId]), GetParam().expected_scalar_value);

    ExpectedValueMetadataSpec expected_scalar_metadata;
    expected_scalar_metadata.value_range = GetParam().expected_scalar_range;
    ExpectProjectionMetadata(value_meta_[kScalarResultId], expected_scalar_metadata);
}

INSTANTIATE_TEST_SUITE_P(
    SubtreeShapes,
    CompositeExtractSubtreeMetadataTests,
    Values(kStructMemberVectorCase,
           kStructMemberArrayCase,
           kNestedStructSubtreeCase),
    [](const TestParamInfo<SubtreeProjectionTestCase>& info) {
        return std::string(info.param.name);
    });

namespace
{

struct StoredLoadProjectionTestCase
{
    const char* name;
    spv::StorageClass storage_class;
    const SubtreeProjectionTestCase* projection;
};

const StoredLoadProjectionTestCase kFunctionVectorMemberCase{
    "FunctionVectorMember",
    spv::StorageClass::StorageClassFunction,
    &kStructMemberVectorCase,
};

const StoredLoadProjectionTestCase kPrivateArrayMemberCase{
    "PrivateArrayMember",
    spv::StorageClass::StorageClassPrivate,
    &kStructMemberArrayCase,
};

const StoredLoadProjectionTestCase kWorkgroupNestedStructCase{
    "WorkgroupNestedStruct",
    spv::StorageClass::StorageClassWorkgroup,
    &kNestedStructSubtreeCase,
};

} // namespace

// restore an aggregate-valued descendant from an ancestor stored
// snapshot, then extract a scalar to verify the projected metadata end to end.
class StoredAggregateLoadProjectionTests : public SPIRVSimulatorMockBase,
                                           public TestWithParam<StoredLoadProjectionTestCase>
{
  public:
    MOCK_METHOD((std::pair<std::byte*, uint64_t>),
                ResolvePointerV,
                (const ::SPIRVSimulator::PointerV& pointer),
                (const, override));
    MOCK_METHOD(std::optional<::SPIRVSimulator::Value>,
                ReadPointer,
                (const ::SPIRVSimulator::PointerV& pointer),
                (override));

  protected:
    static constexpr uint32_t kWholeObjectTypeId = 7100;
    static constexpr uint32_t kAncestorPointerId = 7101;
    static constexpr uint32_t kLoadPointerId = 7102;
    static constexpr uint32_t kStoredValueId = 7103;
    static constexpr uint32_t kLoadResultId = 7104;
    static constexpr uint32_t kScalarResultId = 7105;
    static constexpr uint32_t kMaximumTestId = 7105;

    void SetUp() override
    {
        // Keep optional external tracking and diagnostic paths disabled; this
        // suite only exercises internal stored-value metadata restoration.
        memory_flag_tracker_ = nullptr;
        verbose_ = false;

        if (values_.size() <= kMaximumTestId)
        {
            values_.resize(kMaximumTestId + 1);
        }
        if (value_meta_.size() <= kMaximumTestId)
        {
            value_meta_.resize(kMaximumTestId + 1);
        }

        value_setup_ = MakeSubtreeProjectionValueSetup(GetParam().projection->shape);

        const ::SPIRVSimulator::PointerV ancestor_pointer = MakePointer({});
        const ::SPIRVSimulator::PointerV load_pointer =
            MakePointer(GetParam().projection->selected_path);
        values_[kAncestorPointerId] = ancestor_pointer;
        values_[kLoadPointerId] = load_pointer;

        ::SPIRVSimulator::StoredValueMetadataSnapshot stored_metadata;
        const auto whole_object_sentinel =
            MakeProjectionRange(true, true, 800, 899, 1);
        stored_metadata.value_range = whole_object_sentinel;
        stored_metadata.component_ranges = { whole_object_sentinel };
        stored_metadata.subobject_scalar_ranges =
            GetParam().projection->source_subobject_ranges;
        stored_metadata.address_range_valid = true;
        stored_metadata.address_range_min = 800;
        stored_metadata.address_range_max = 899;
        stored_metadata.address_range_stride = 1;
        stored_metadata.address_element_size = 8;

        values_stored_[kAncestorPointerId] = ::SPIRVSimulator::StoredValueRecord{
            kStoredValueId,
            ancestor_pointer,
            stored_metadata,
        };

        EXPECT_CALL(*this, GetValue(_))
            .WillRepeatedly([this](uint32_t id) -> ::SPIRVSimulator::Value& {
                return values_[id];
            });
        EXPECT_CALL(*this, SetValue(_, _, true))
            .Times(2)
            .WillRepeatedly([this](uint32_t id,
                                   const ::SPIRVSimulator::Value& value,
                                   bool clear_meta) {
                ::SPIRVSimulator::SPIRVSimulator::SetValue(id, value, clear_meta);
            });
        EXPECT_CALL(*this, GetTypeByResultId(kLoadResultId))
            .WillRepeatedly(ReturnRef(value_setup_.extracted_type));
        EXPECT_CALL(*this, GetTypeByResultId(kScalarResultId))
            .WillRepeatedly(ReturnRef(types_[CommonTypes::u64]));
        EXPECT_CALL(*this, ResolvePointerV(_))
            .Times(1)
            .WillOnce(Return(std::pair<std::byte*, uint64_t>{ backing_.data(), 0 }));
        EXPECT_CALL(*this, ReadPointer(_))
            .Times(1)
            .WillOnce(Return(std::optional<::SPIRVSimulator::Value>{ value_setup_.extracted_value }));
        EXPECT_CALL(*this, TransferFlags(A<uint32_t>(), A<uint32_t>()))
            .Times(1);
    }

    ::SPIRVSimulator::PointerV MakePointer(std::vector<uint32_t> idx_path) const
    {
        return ::SPIRVSimulator::PointerV{
            1,
            0,
            kWholeObjectTypeId,
            kAncestorPointerId,
            static_cast<uint32_t>(GetParam().storage_class),
            std::move(idx_path),
        };
    }

    bool ExecuteLoad()
    {
        std::vector<uint32_t> words{
            static_cast<uint32_t>(spv::Op::OpLoad),
            value_setup_.extracted_type_id,
            kLoadResultId,
            kLoadPointerId,
        };
        const ::SPIRVSimulator::Instruction instruction{
            .opcode = spv::Op::OpLoad,
            .word_count = static_cast<uint16_t>(words.size()),
            .words = words,
        };
        return ExecuteInstruction(instruction);
    }

    bool ExecuteScalarExtract()
    {
        std::vector<uint32_t> words{
            static_cast<uint32_t>(spv::Op::OpCompositeExtract),
            CommonTypes::u64,
            kScalarResultId,
            kLoadResultId,
        };
        words.insert(words.end(),
                     GetParam().projection->followup_scalar_path.begin(),
                     GetParam().projection->followup_scalar_path.end());

        const ::SPIRVSimulator::Instruction instruction{
            .opcode = spv::Op::OpCompositeExtract,
            .word_count = static_cast<uint16_t>(words.size()),
            .words = words,
        };
        return ExecuteInstruction(instruction);
    }

    std::array<std::byte, 1> backing_{};
    SubtreeProjectionValueSetup value_setup_;
};

TEST_P(StoredAggregateLoadProjectionTests, ProjectsAncestorSnapshotIntoLoadedSubobject)
{
    SeedStaleProjectionMetadata(value_meta_[kLoadResultId]);
    SeedStaleProjectionMetadata(value_meta_[kScalarResultId]);

    ASSERT_TRUE(ExecuteLoad());

    ExpectProjectedConcreteValue(GetParam().projection->shape,
                                 values_[kLoadResultId]);
    ExpectProjectionMetadata(value_meta_[kLoadResultId],
                             GetParam().projection->expected_extracted_metadata);
    EXPECT_TRUE(values_stored_.contains(kAncestorPointerId));
    EXPECT_FALSE(values_stored_.contains(kLoadPointerId));

    ASSERT_TRUE(ExecuteScalarExtract());

    ASSERT_TRUE(std::holds_alternative<uint64_t>(values_[kScalarResultId]));
    EXPECT_EQ(std::get<uint64_t>(values_[kScalarResultId]),
              GetParam().projection->expected_scalar_value);

    ExpectedValueMetadataSpec expected_scalar_metadata;
    expected_scalar_metadata.value_range =
        GetParam().projection->expected_scalar_range;
    ExpectProjectionMetadata(value_meta_[kScalarResultId], expected_scalar_metadata);
}

INSTANTIATE_TEST_SUITE_P(
    InternalStorageClasses,
    StoredAggregateLoadProjectionTests,
    Values(kFunctionVectorMemberCase,
           kPrivateArrayMemberCase,
           kWorkgroupNestedStructCase),
    [](const TestParamInfo<StoredLoadProjectionTestCase>& info) {
        return std::string(info.param.name);
    });

// Helper-level coverage for ancestor/descendant overlap invalidation in both
// stored-value maps. These tests do not execute OpStore or OpLoad.
class StoredValueInvalidationTests : public SPIRVSimulatorMockBase, public Test
{
  protected:
    static ::SPIRVSimulator::PointerV MakePointer(uint64_t pointer_handle,
                                                  std::vector<uint32_t> idx_path)
    {
        return ::SPIRVSimulator::PointerV{
            pointer_handle,
            0,
            100,
            200,
            spv::StorageClass::StorageClassFunction,
            std::move(idx_path),
        };
    }

    static ::SPIRVSimulator::PointerLocationKey MakeLocationKey(
        const ::SPIRVSimulator::PointerV& pointer,
        uint64_t byte_offset)
    {
        return ::SPIRVSimulator::PointerLocationKey{
            pointer.storage_class,
            pointer.base_result_id,
            pointer.pointer_handle,
            byte_offset,
            pointer.idx_path,
        };
    }

    void AddStoredValue(uint32_t pointer_id,
                        uint32_t stored_result_id,
                        const ::SPIRVSimulator::PointerV& pointer,
                        uint64_t byte_offset)
    {
        ::SPIRVSimulator::StoredValueRecord record{ stored_result_id, pointer, {} };
        values_stored_[pointer_id] = record;
        values_stored_by_memory_location_[MakeLocationKey(pointer, byte_offset)] = record;
    }

    void Invalidate(const ::SPIRVSimulator::PointerV& pointer)
    {
        InvalidateOverlappingStoredValues(pointer);
    }

    bool HasPointerRecord(uint32_t pointer_id) const
    {
        return values_stored_.contains(pointer_id);
    }

    bool HasLocationRecord(const ::SPIRVSimulator::PointerV& pointer,
                           uint64_t byte_offset) const
    {
        return values_stored_by_memory_location_.contains(
            MakeLocationKey(pointer, byte_offset));
    }
};

TEST_F(StoredValueInvalidationTests, WholeObjectTargetInvalidatesSubobjectRecordsInBothMaps)
{
    const auto whole_object = MakePointer(1, {});
    const auto first_member = MakePointer(1, { 0 });
    const auto nested_member = MakePointer(1, { 1, 0 });
    const auto unrelated_member = MakePointer(2, { 0 });

    AddStoredValue(10, 100, first_member, 4);
    AddStoredValue(11, 101, nested_member, 8);
    AddStoredValue(12, 102, unrelated_member, 4);

    Invalidate(whole_object);

    EXPECT_FALSE(HasPointerRecord(10));
    EXPECT_FALSE(HasLocationRecord(first_member, 4));
    EXPECT_FALSE(HasPointerRecord(11));
    EXPECT_FALSE(HasLocationRecord(nested_member, 8));
    EXPECT_TRUE(HasPointerRecord(12));
    EXPECT_TRUE(HasLocationRecord(unrelated_member, 4));
}

TEST_F(StoredValueInvalidationTests, SubobjectTargetInvalidatesAncestorAndPreservesSiblingInBothMaps)
{
    const auto whole_object = MakePointer(1, {});
    const auto first_member = MakePointer(1, { 0 });
    const auto first_member_leaf = MakePointer(1, { 0, 1 });
    const auto sibling_member = MakePointer(1, { 1 });

    AddStoredValue(20, 200, whole_object, 0);
    AddStoredValue(21, 201, first_member_leaf, 4);
    AddStoredValue(22, 202, sibling_member, 8);

    Invalidate(first_member);

    EXPECT_FALSE(HasPointerRecord(20));
    EXPECT_FALSE(HasLocationRecord(whole_object, 0));
    EXPECT_FALSE(HasPointerRecord(21));
    EXPECT_FALSE(HasLocationRecord(first_member_leaf, 4));
    EXPECT_TRUE(HasPointerRecord(22));
    EXPECT_TRUE(HasLocationRecord(sibling_member, 8));
}


namespace
{

enum class StoredTarget
{
    WholeObject,
    FirstMember,
    SecondMember,
};

::SPIRVSimulator::DenseValueRangeInfo MakeRange(uint64_t min, uint64_t max)
{
    return ::SPIRVSimulator::DenseValueRangeInfo{ true, true, true, min, max, 1 };
}

} // namespace

// shared integration environment for store/load ordering.
// The fixture mocks concrete Pair memory, executes the real OpStore and OpLoad
// paths, and never pre-populates either stored-value map.
class StoreLoadMetadataTests : public SPIRVSimulatorMockBase, public Test
{
  public:
    MOCK_METHOD((std::pair<std::byte*, uint64_t>),
                ResolvePointerV,
                (const ::SPIRVSimulator::PointerV& pointer),
                (const, override));
    MOCK_METHOD(bool,
                WritePointer,
                (const ::SPIRVSimulator::PointerV& pointer,
                 const ::SPIRVSimulator::Value& value),
                (override));
    MOCK_METHOD(std::optional<::SPIRVSimulator::Value>,
                ReadPointer,
                (const ::SPIRVSimulator::PointerV& pointer),
                (override));
    MOCK_METHOD((std::vector<::SPIRVSimulator::DataSourceBits>),
                FindDataSourcesFromResultID,
                (uint32_t result_id, uint32_t* property_flags),
                (override));

  protected:
    static constexpr uint32_t kPairTypeId = 6000;
    static constexpr uint32_t kWholePointerId = 6001;
    static constexpr uint32_t kFirstMemberPointerId = 6002;
    static constexpr uint32_t kSecondMemberPointerId = 6003;
    static constexpr uint32_t kStoreValueBaseId = 6010;
    static constexpr uint32_t kLoadResultBaseId = 6020;
    static constexpr uint32_t kMaximumTestId = 6030;

    void SetUp() override
    {
        // Direct opcode execution has no Run() call stack and no external
        // tracker, so keep those optional production paths disabled.
        memory_flag_tracker_ = nullptr;
        verbose_ = false;

        if (values_.size() <= kMaximumTestId)
        {
            values_.resize(kMaximumTestId + 1);
        }
        if (value_meta_.size() <= kMaximumTestId)
        {
            value_meta_.resize(kMaximumTestId + 1);
        }

        types_[kPairTypeId] = ::SPIRVSimulator::Type::Struct(kPairTypeId);

        values_[kWholePointerId] = MakePointer({});
        values_[kFirstMemberPointerId] = MakePointer({ 0 });
        values_[kSecondMemberPointerId] = MakePointer({ 1 });

        EXPECT_CALL(*this, GetValue(_))
            .WillRepeatedly([this](uint32_t id) -> ::SPIRVSimulator::Value& {
                return values_[id];
            });
        EXPECT_CALL(*this, SetValue(_, _, _))
            .WillRepeatedly([this](uint32_t id,
                                   const ::SPIRVSimulator::Value& value,
                                   bool clear_meta) {
                ++set_value_count_;
                SetValueWithBaseImplementation(id, value, clear_meta);
            });
        EXPECT_CALL(*this, GetTypeByResultId(_))
            .WillRepeatedly(ReturnRef(types_[CommonTypes::u64]));
        EXPECT_CALL(*this, ResolvePointerV(_))
            .WillRepeatedly([this](const ::SPIRVSimulator::PointerV& pointer) {
                return std::pair<std::byte*, uint64_t>{ backing_.data(), GetByteOffset(pointer) };
            });
        EXPECT_CALL(*this, WritePointer(_, _))
            .WillRepeatedly([this](const ::SPIRVSimulator::PointerV& pointer,
                                   const ::SPIRVSimulator::Value& value) -> bool {
                ++write_count_;
                WriteConcreteValue(pointer, value);
                return true;
            });
        EXPECT_CALL(*this, ReadPointer(_))
            .WillRepeatedly([this](const ::SPIRVSimulator::PointerV& pointer)
                                -> std::optional<::SPIRVSimulator::Value> {
                ++read_count_;
                return ReadConcreteValue(pointer);
            });
        EXPECT_CALL(*this, FindDataSourcesFromResultID(_, _))
            .WillRepeatedly([](uint32_t, uint32_t* property_flags) {
                if (property_flags != nullptr)
                {
                    *property_flags = 0;
                }
                return std::vector<::SPIRVSimulator::DataSourceBits>{};
            });
    }

    void SetValueWithBaseImplementation(uint32_t id,
                                        const ::SPIRVSimulator::Value& value,
                                        bool clear_meta)
    {
        ::SPIRVSimulator::SPIRVSimulator::SetValue(id, value, clear_meta);
    }

    ::SPIRVSimulator::PointerV MakePointer(std::vector<uint32_t> idx_path) const
    {
        return ::SPIRVSimulator::PointerV{
            1,
            0,
            kPairTypeId,
            kWholePointerId,
            spv::StorageClass::StorageClassFunction,
            std::move(idx_path),
        };
    }

    static uint64_t GetByteOffset(const ::SPIRVSimulator::PointerV& pointer)
    {
        if (pointer.idx_path == std::vector<uint32_t>{ 1 })
        {
            return sizeof(uint64_t);
        }
        return 0;
    }

    static uint32_t GetPointerId(StoredTarget target)
    {
        switch (target)
        {
            case StoredTarget::WholeObject:
                return kWholePointerId;
            case StoredTarget::FirstMember:
                return kFirstMemberPointerId;
            case StoredTarget::SecondMember:
                return kSecondMemberPointerId;
        }
        return 0;
    }

    void WriteConcreteValue(const ::SPIRVSimulator::PointerV& pointer,
                            const ::SPIRVSimulator::Value& value)
    {
        if (pointer.idx_path.empty())
        {
            const auto& aggregate = *std::get<std::shared_ptr<::SPIRVSimulator::AggregateV>>(value);
            pair_values_[0] = std::get<uint64_t>(aggregate.elems[0]);
            pair_values_[1] = std::get<uint64_t>(aggregate.elems[1]);
            return;
        }

        pair_values_[pointer.idx_path[0]] = std::get<uint64_t>(value);
    }

    ::SPIRVSimulator::Value ReadConcreteValue(const ::SPIRVSimulator::PointerV& pointer) const
    {
        if (pointer.idx_path.empty())
        {
            return std::make_shared<::SPIRVSimulator::AggregateV>(
                std::initializer_list<::SPIRVSimulator::Value>{ pair_values_[0], pair_values_[1] });
        }

        return pair_values_[pointer.idx_path[0]];
    }

    void ExecuteStore(StoredTarget target, uint32_t value_id)
    {
        std::vector<uint32_t> words = { 0, GetPointerId(target), value_id };
        const ::SPIRVSimulator::Instruction instruction{
            .opcode = spv::Op::OpStore,
            .word_count = static_cast<uint16_t>(words.size()),
            .words = words,
        };
        Op_Store(instruction);
    }

    uint32_t ExecuteLoad(StoredTarget target, uint32_t type_id)
    {
        const uint32_t result_id = next_load_result_id_++;
        std::vector<uint32_t> words = {
            0,
            type_id,
            result_id,
            GetPointerId(target),
        };
        const ::SPIRVSimulator::Instruction instruction{
            .opcode = spv::Op::OpLoad,
            .word_count = static_cast<uint16_t>(words.size()),
            .words = words,
        };
        Op_Load(instruction);
        return result_id;
    }

    void StoreScalar(StoredTarget target,
                     uint64_t value,
                     const ::SPIRVSimulator::DenseValueRangeInfo& range)
    {
        const uint32_t value_id = next_store_value_id_++;
        values_[value_id] = value;
        value_meta_[value_id] = {};
        value_meta_[value_id].value_range = range;
        ExecuteStore(target, value_id);
    }

    void StorePair(uint64_t first_value,
                   uint64_t second_value,
                   const ::SPIRVSimulator::DenseValueRangeInfo& first_range,
                   const ::SPIRVSimulator::DenseValueRangeInfo& second_range,
                   bool include_top_level_sentinels = false)
    {
        const uint32_t value_id = next_store_value_id_++;
        values_[value_id] = std::make_shared<::SPIRVSimulator::AggregateV>(
            std::initializer_list<::SPIRVSimulator::Value>{ first_value, second_value });

        ::SPIRVSimulator::ValueMetadata& metadata = value_meta_[value_id];
        metadata = {};
        metadata.subobject_scalar_ranges = {
            ::SPIRVSimulator::SubobjectScalarRangeEntry{ { 0 }, first_range },
            ::SPIRVSimulator::SubobjectScalarRangeEntry{ { 1 }, second_range },
        };
        if (include_top_level_sentinels)
        {
            metadata.value_range = MakeRange(900, 909);
            metadata.component_ranges = { first_range, second_range };
        }

        ExecuteStore(StoredTarget::WholeObject, value_id);
    }

    uint32_t LoadScalar(StoredTarget target)
    {
        return ExecuteLoad(target, CommonTypes::u64);
    }

    uint32_t LoadPair()
    {
        return ExecuteLoad(StoredTarget::WholeObject, kPairTypeId);
    }

    static void ExpectRange(const ::SPIRVSimulator::DenseValueRangeInfo& actual,
                            const ::SPIRVSimulator::DenseValueRangeInfo& expected)
    {
        EXPECT_EQ(actual.valid, expected.valid);
        EXPECT_EQ(actual.thread_dependent, expected.thread_dependent);
        EXPECT_EQ(actual.dense_range, expected.dense_range);
        EXPECT_EQ(actual.min, expected.min);
        EXPECT_EQ(actual.max, expected.max);
        EXPECT_EQ(actual.stride, expected.stride);
    }

    void ExpectScalarResult(uint32_t result_id,
                            uint64_t expected_value,
                            const ::SPIRVSimulator::DenseValueRangeInfo& expected_range) const
    {
        ASSERT_TRUE(std::holds_alternative<uint64_t>(values_[result_id]));
        EXPECT_EQ(std::get<uint64_t>(values_[result_id]), expected_value);

        const ::SPIRVSimulator::ValueMetadata& metadata = value_meta_[result_id];
        ExpectRange(metadata.value_range, expected_range);
        EXPECT_TRUE(metadata.component_ranges.empty());
        EXPECT_TRUE(metadata.subobject_scalar_ranges.empty());
        EXPECT_FALSE(metadata.address_range_valid);
    }

    void ExpectPairResultWithoutMetadata(uint32_t result_id,
                                         uint64_t expected_first,
                                         uint64_t expected_second) const
    {
        ASSERT_TRUE(std::holds_alternative<std::shared_ptr<::SPIRVSimulator::AggregateV>>(
            values_[result_id]));
        const auto& aggregate = *std::get<std::shared_ptr<::SPIRVSimulator::AggregateV>>(
            values_[result_id]);
        ASSERT_EQ(aggregate.elems.size(), 2u);
        EXPECT_EQ(std::get<uint64_t>(aggregate.elems[0]), expected_first);
        EXPECT_EQ(std::get<uint64_t>(aggregate.elems[1]), expected_second);

        const ::SPIRVSimulator::ValueMetadata& metadata = value_meta_[result_id];
        ExpectRange(metadata.value_range, {});
        EXPECT_TRUE(metadata.component_ranges.empty());
        EXPECT_TRUE(metadata.subobject_scalar_ranges.empty());
        EXPECT_FALSE(metadata.address_range_valid);
    }

    void ExpectOperationCounts(size_t expected_stores, size_t expected_loads) const
    {
        EXPECT_EQ(write_count_, expected_stores);
        EXPECT_EQ(read_count_, expected_loads);
        EXPECT_EQ(set_value_count_, expected_loads);
    }

    std::array<std::byte, 16> backing_{};
    std::array<uint64_t, 2> pair_values_{};
    uint32_t next_store_value_id_ = kStoreValueBaseId;
    uint32_t next_load_result_id_ = kLoadResultBaseId;
    size_t write_count_ = 0;
    size_t read_count_ = 0;
    size_t set_value_count_ = 0;
};

// a later whole-object store must replace stale metadata from an
// earlier descendant store, and the scalar load must project the new leaf range.
TEST_F(StoreLoadMetadataTests, WholeObjectStoreReplacesDescendantMetadata)
{
    StoreScalar(StoredTarget::FirstMember, 11, MakeRange(10, 19));
    StorePair(101, 202, MakeRange(100, 109), MakeRange(200, 209));

    const uint32_t result_id = LoadScalar(StoredTarget::FirstMember);

    ExpectScalarResult(result_id, 101, MakeRange(100, 109));
    ExpectOperationCounts(2, 1);
}

// under the current conservative policy, a descendant store drops
// the old ancestor snapshot but remains available to an exact descendant load.
TEST_F(StoreLoadMetadataTests, DescendantStoreDropsAncestorSnapshot)
{
    StorePair(11, 22, MakeRange(10, 19), MakeRange(20, 29), true);
    StoreScalar(StoredTarget::FirstMember, 31, MakeRange(30, 39));

    const uint32_t whole_result_id = LoadPair();
    const uint32_t member_result_id = LoadScalar(StoredTarget::FirstMember);

    ExpectPairResultWithoutMetadata(whole_result_id, 31, 22);
    ExpectScalarResult(member_result_id, 31, MakeRange(30, 39));
    ExpectOperationCounts(2, 2);
}

// stores to non-overlapping sibling paths must retain independent
// reaching metadata for their subsequent exact loads.
TEST_F(StoreLoadMetadataTests, SiblingStoresRestoreIndependentMetadata)
{
    StoreScalar(StoredTarget::FirstMember, 41, MakeRange(40, 49));
    StoreScalar(StoredTarget::SecondMember, 52, MakeRange(50, 59));

    const uint32_t first_result_id = LoadScalar(StoredTarget::FirstMember);
    const uint32_t second_result_id = LoadScalar(StoredTarget::SecondMember);

    ExpectScalarResult(first_result_id, 41, MakeRange(40, 49));
    ExpectScalarResult(second_result_id, 52, MakeRange(50, 59));
    ExpectOperationCounts(2, 2);
}
