#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "spirv_simulator.hpp"
#include "testing_common.hpp"

using namespace testing;

class AccessChainAddressRangeTests : public SPIRVSimulatorMockBase, public Test
{
  public:
    MOCK_METHOD((std::pair<std::byte*, uint64_t>),
                ResolvePointerV,
                (const ::SPIRVSimulator::PointerV& pointer),
                (const, override));
    MOCK_METHOD(size_t, GetBitsizeOfType, (uint32_t type_id), (const, override));
    MOCK_METHOD(uint32_t,
                GetTargetPointerType,
                (const ::SPIRVSimulator::PointerV& pointer),
                (const, override));
    MOCK_METHOD(bool,
                WritePointer,
                (const ::SPIRVSimulator::PointerV& pointer, const ::SPIRVSimulator::Value& value),
                (override));
    MOCK_METHOD(void, OverrideFlagsPointee, (uint32_t pointer_id, uint32_t result_id), (override));
    MOCK_METHOD((std::vector<::SPIRVSimulator::DataSourceBits>),
                FindDataSourcesFromResultID,
                (uint32_t result_id, uint32_t* property_flags),
                (override));
    MOCK_METHOD(bool,
                PointeeValueIsDescriptorBuffer,
                (const ::SPIRVSimulator::PointerV& pointer),
                (const, override));
    MOCK_METHOD(bool, ValueHoldsPbufferPtr, (uint32_t result_id), (const, override));
    MOCK_METHOD(bool, ValueIsArbitrary, (uint32_t result_id), (const, override));

  protected:
    bool TrySetAddressRange(const ::SPIRVSimulator::Instruction& instruction,
                            const ::SPIRVSimulator::PointerV& result_pointer)
    {
        return TrySetDenseAccessChainAddressRange(instruction, result_pointer);
    }

    void SetFinalIndexRange(uint32_t result_id, uint64_t min, uint64_t max, uint64_t stride = 1)
    {
        value_meta_.resize(std::max<size_t>(value_meta_.size(), result_id + 1));
        value_meta_[result_id].value_range =
            ::SPIRVSimulator::DenseValueRangeInfo{ true, true, true, min, max, stride };
    }

    const ::SPIRVSimulator::ValueMetadata& GetMetadata(uint32_t result_id) const
    {
        return value_meta_[result_id];
    }

    void SetAddressRange(uint32_t pointer_id,
                         uint64_t min,
                         uint64_t max,
                         uint64_t stride,
                         uint64_t element_size)
    {
        value_meta_.resize(std::max<size_t>(value_meta_.size(), pointer_id + 1));
        ::SPIRVSimulator::ValueMetadata& metadata = value_meta_[pointer_id];
        metadata.address_range_valid  = true;
        metadata.address_range_min    = min;
        metadata.address_range_max    = max;
        metadata.address_range_stride = stride;
        metadata.address_element_size = element_size;
    }

    bool TryGetStoreRange(const ::SPIRVSimulator::PointerV& pointer,
                          uint32_t pointer_id,
                          uint32_t result_id,
                          uint64_t& dst_start,
                          uint64_t& byte_size,
                          uint64_t& element_size)
    {
        return TryGetDenseStoreRange(
            pointer, pointer_id, result_id, dst_start, byte_size, element_size);
    }

    void Store(const ::SPIRVSimulator::Instruction& instruction)
    {
        Op_Store(instruction);
    }

    void SetRuntimeState(::SPIRVSimulator::MemoryFlagTracker* memory_flag_tracker,
                         ::SPIRVSimulator::SimulationResults* simulation_results)
    {
        memory_flag_tracker_ = memory_flag_tracker;
        simulation_results_  = simulation_results;
    }

    std::array<std::byte, 64> backing_{};
};

enum class FinalIndexContainerKind
{
    RuntimeArray,
    FixedArray,
    Vector,
};

enum class FinalIndexResultKind
{
    U32,
    U64,
    F32,
    PhysicalPointer,
};

struct FinalDynamicIndexCase
{
    const char*             name;
    FinalIndexContainerKind container_kind;
    FinalIndexResultKind    result_kind;
    uint64_t                index_min;
    uint64_t                index_max;
    uint64_t                index_stride;
    uint64_t                byte_step;
    uint64_t                element_size;
};

// Verifies that a supported final dynamic index publishes the exact dense range
// across accepted scalar or pointer, array or vector, and single or multi-element cases.
class FinalDynamicIndexAddressRangeTests
    : public AccessChainAddressRangeTests,
      public WithParamInterface<FinalDynamicIndexCase>
{};

TEST_P(FinalDynamicIndexAddressRangeTests, ProducesDenseRange)
{
    constexpr uint32_t result_type_id           = 5000;
    constexpr uint32_t result_id                = 5001;
    constexpr uint32_t base_id                  = 5002;
    constexpr uint32_t final_index_id           = 5003;
    constexpr uint32_t container_type_id        = 5004;
    constexpr uint32_t physical_pointer_type_id = 5005;
    constexpr uint32_t array_length_id           = 5006;

    const FinalDynamicIndexCase& test_case = GetParam();

    uint32_t result_pointee_type_id = CommonTypes::u64;
    switch (test_case.result_kind)
    {
        case FinalIndexResultKind::U32:
            result_pointee_type_id = CommonTypes::u32;
            break;
        case FinalIndexResultKind::U64:
            result_pointee_type_id = CommonTypes::u64;
            break;
        case FinalIndexResultKind::F32:
            result_pointee_type_id = CommonTypes::f32;
            break;
        case FinalIndexResultKind::PhysicalPointer:
            result_pointee_type_id = physical_pointer_type_id;
            break;
    }

    const ::SPIRVSimulator::Type physical_pointer_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, CommonTypes::u64);
    if (test_case.result_kind == FinalIndexResultKind::PhysicalPointer)
    {
        EXPECT_CALL(*this, GetTypeByTypeId(physical_pointer_type_id))
            .WillRepeatedly(ReturnRef(physical_pointer_type));
    }

    const ::SPIRVSimulator::Type result_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, result_pointee_type_id);

    ::SPIRVSimulator::Type container_type;
    switch (test_case.container_kind)
    {
        case FinalIndexContainerKind::RuntimeArray:
            container_type = ::SPIRVSimulator::Type::RuntimeArray(result_pointee_type_id);
            break;
        case FinalIndexContainerKind::FixedArray:
            container_type =
                ::SPIRVSimulator::Type::Array(result_pointee_type_id, array_length_id);
            break;
        case FinalIndexContainerKind::Vector:
            container_type = ::SPIRVSimulator::Type::Vector(result_pointee_type_id, 4);
            break;
    }

    EXPECT_CALL(*this, GetTypeByTypeId(result_type_id)).WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByTypeId(container_type_id)).WillRepeatedly(ReturnRef(container_type));
    EXPECT_CALL(*this, GetTargetPointerType(_)).WillOnce(Return(container_type_id));
    EXPECT_CALL(*this, GetBitsizeOfType(result_pointee_type_id))
        .WillOnce(Return(test_case.element_size * 8));
    EXPECT_CALL(*this, ResolvePointerV(_))
        .WillRepeatedly([this, &test_case](const ::SPIRVSimulator::PointerV& pointer) {
            return std::pair<std::byte*, uint64_t>{
                backing_.data(), pointer.idx_path.back() * test_case.byte_step
            };
        });

    SetFinalIndexRange(final_index_id,
                       test_case.index_min,
                       test_case.index_max,
                       test_case.index_stride);
    std::vector<uint32_t> words = { 0, result_type_id, result_id, base_id, final_index_id };
    const ::SPIRVSimulator::Instruction instruction{
        .opcode = spv::Op::OpAccessChain,
        .word_count = static_cast<uint16_t>(words.size()),
        .words = words,
    };
    const ::SPIRVSimulator::PointerV result_pointer{
        .storage_class = spv::StorageClassPhysicalStorageBuffer,
        .idx_path = { 0 },
    };

    EXPECT_TRUE(TrySetAddressRange(instruction, result_pointer));

    const uint64_t base_address = ::SPIRVSimulator::bit_cast<uint64_t>(backing_.data());
    const ::SPIRVSimulator::ValueMetadata& metadata = GetMetadata(result_id);
    EXPECT_TRUE(metadata.address_range_valid);
    EXPECT_EQ(metadata.address_range_min,
              base_address + test_case.index_min * test_case.byte_step);
    EXPECT_EQ(metadata.address_range_max,
              base_address + test_case.index_max * test_case.byte_step);
    EXPECT_EQ(metadata.address_range_stride, test_case.element_size);
    EXPECT_EQ(metadata.address_element_size, test_case.element_size);
    EXPECT_NE(metadata.flags & SPS_FLAG_THREAD_SPECIFIC, 0u);
}

INSTANTIATE_TEST_SUITE_P(
    SupportedFinalIndexCases,
    FinalDynamicIndexAddressRangeTests,
    Values(FinalDynamicIndexCase{ "U64RuntimeArray",
                                  FinalIndexContainerKind::RuntimeArray,
                                  FinalIndexResultKind::U64,
                                  0, 2, 1, 8, 8 },
           FinalDynamicIndexCase{ "U32FixedArrayInBounds",
                                  FinalIndexContainerKind::FixedArray,
                                  FinalIndexResultKind::U32,
                                  1, 3, 1, 4, 4 },
           FinalDynamicIndexCase{ "F32RuntimeArray",
                                  FinalIndexContainerKind::RuntimeArray,
                                  FinalIndexResultKind::F32,
                                  2, 4, 1, 4, 4 },
           FinalDynamicIndexCase{ "PhysicalPointerRuntimeArray",
                                  FinalIndexContainerKind::RuntimeArray,
                                  FinalIndexResultKind::PhysicalPointer,
                                  0, 2, 1, 8, 8 },
           FinalDynamicIndexCase{ "U64VectorInBounds",
                                  FinalIndexContainerKind::Vector,
                                  FinalIndexResultKind::U64,
                                  1, 3, 1, 8, 8 },
           FinalDynamicIndexCase{ "U64RuntimeArraySingleElementTightlyPacked",
                                  FinalIndexContainerKind::RuntimeArray,
                                  FinalIndexResultKind::U64,
                                  0, 0, 1, 8, 8 },
           FinalDynamicIndexCase{ "U64RuntimeArraySingleElementPadded",
                                  FinalIndexContainerKind::RuntimeArray,
                                  FinalIndexResultKind::U64,
                                  1, 1, 1, 16, 8 },
           FinalDynamicIndexCase{ "PhysicalPointerRuntimeArraySingleElementPadded",
                                  FinalIndexContainerKind::RuntimeArray,
                                  FinalIndexResultKind::PhysicalPointer,
                                  2, 2, 1, 16, 8 },
           FinalDynamicIndexCase{ "U64VectorSingleComponent",
                                  FinalIndexContainerKind::Vector,
                                  FinalIndexResultKind::U64,
                                  2, 2, 1, 8, 8 }),
    [](const TestParamInfo<FinalDynamicIndexCase>& info) { return std::string(info.param.name); });

// Verifies that OpStore includes the final element when consuming valid dense
// address-range metadata and records the full span without requesting fallback.
TEST_F(AccessChainAddressRangeTests, DenseAddressRangeIsConsumedByStore)
{
    constexpr uint32_t pointer_id = 5050;
    constexpr uint32_t result_id  = 5051;

    ::SPIRVSimulator::MemoryFlagTracker memory_flag_tracker;
    ::SPIRVSimulator::SimulationResults simulation_results;
    SetRuntimeState(&memory_flag_tracker, &simulation_results);

    const ::SPIRVSimulator::PointerV pointer{
        .base_result_id = pointer_id,
        .storage_class = spv::StorageClassPhysicalStorageBuffer,
        .idx_path = { 0 },
    };
    ::SPIRVSimulator::Value pointer_value = pointer;
    ::SPIRVSimulator::Value stored_value  = uint64_t{ 0x1234 };

    const uint64_t range_start = ::SPIRVSimulator::bit_cast<uint64_t>(backing_.data());
    SetAddressRange(pointer_id, range_start, range_start + 16, 8, 8);
    value_meta_.resize(std::max<size_t>(value_meta_.size(), result_id + 1));
    value_meta_[result_id].flags = SPS_FLAG_THREAD_SPECIFIC;

    EXPECT_CALL(*this, ResolvePointerV(_))
        .WillRepeatedly([this](const ::SPIRVSimulator::PointerV&) {
            return std::pair<std::byte*, uint64_t>{ backing_.data(), 0 };
        });

    uint64_t dense_dst_start    = 0;
    uint64_t dense_byte_size    = 0;
    uint64_t dense_element_size = 0;
    ASSERT_TRUE(TryGetStoreRange(pointer,
                                 pointer_id,
                                 result_id,
                                 dense_dst_start,
                                 dense_byte_size,
                                 dense_element_size));
    EXPECT_EQ(dense_dst_start, range_start);
    EXPECT_EQ(dense_byte_size, 24u);
    EXPECT_EQ(dense_element_size, 8u);

    EXPECT_CALL(*this, GetValue(pointer_id)).WillRepeatedly(ReturnRef(pointer_value));
    EXPECT_CALL(*this, GetValue(result_id)).WillRepeatedly(ReturnRef(stored_value));
    EXPECT_CALL(*this, PointeeValueIsDescriptorBuffer(_)).WillOnce(Return(false));
    EXPECT_CALL(*this, WritePointer(_, _)).WillOnce(Return(true));
    EXPECT_CALL(*this, OverrideFlagsPointee(pointer_id, result_id)).Times(1);
    EXPECT_CALL(*this, FindDataSourcesFromResultID(result_id, _))
        .WillOnce([](uint32_t, uint32_t* property_flags) {
            *property_flags = 0;
            return std::vector<::SPIRVSimulator::DataSourceBits>{};
        });
    EXPECT_CALL(*this, ValueHoldsPbufferPtr(result_id)).WillOnce(Return(false));
    EXPECT_CALL(*this, ValueIsArbitrary(result_id)).WillOnce(Return(false));

    std::vector<uint32_t> words = { 0, pointer_id, result_id };
    const ::SPIRVSimulator::Instruction instruction{
        .opcode = spv::Op::OpStore,
        .word_count = static_cast<uint16_t>(words.size()),
        .words = words,
    };
    Store(instruction);

    const std::optional<::SPIRVSimulator::MemoryFlagTracker::UniformDerivedRange> tracked_range =
        memory_flag_tracker.queryUniformDerivedRange(range_start);
    ASSERT_TRUE(tracked_range.has_value());
    EXPECT_EQ(tracked_range->start, range_start);
    EXPECT_EQ(tracked_range->end, range_start + 24);
    EXPECT_EQ(tracked_range->element_size, 8u);
    EXPECT_EQ(tracked_range->flags,
              SPS_FLAG_THREAD_SPECIFIC | SPS_FLAG_UNIFORM_DERIVED_RANGE);
    EXPECT_FALSE(simulation_results.full_dispatch_needed);
}

enum class UnsupportedAccessChainCapability
{
    NonFinalDynamicIndex,
    MultipleDynamicIndexes,
    BaseHasAddressRange,
    BaseIsThreadSpecific,
    UniformDescriptorArray,
    UniformConstantDescriptorArray,
    StorageBufferDescriptorArray,
};

struct UnsupportedAccessChainCapabilityCase
{
    const char*                            name;
    UnsupportedAccessChainCapability       capability;
};

// Verifies that unsupported dynamic-index placement, pre-existing base
// contributions, and descriptor-array indirection never publish a partial range.
class UnsupportedAccessChainAddressRangeCapabilityTests
    : public AccessChainAddressRangeTests,
      public WithParamInterface<UnsupportedAccessChainCapabilityCase>
{};

TEST_P(UnsupportedAccessChainAddressRangeCapabilityTests, RejectsAddressRange)
{
    constexpr uint32_t result_type_id           = 5070;
    constexpr uint32_t result_id                = 5071;
    constexpr uint32_t base_id                  = 5072;
    constexpr uint32_t first_index_id           = 5073;
    constexpr uint32_t final_index_id           = 5074;
    constexpr uint32_t container_type_id        = 5075;

    const UnsupportedAccessChainCapabilityCase& test_case = GetParam();

    uint32_t storage_class = spv::StorageClassPhysicalStorageBuffer;
    std::vector<uint32_t> index_ids;
    switch (test_case.capability)
    {
        case UnsupportedAccessChainCapability::NonFinalDynamicIndex:
            SetFinalIndexRange(first_index_id, 0, 2);
            index_ids = { first_index_id, final_index_id };
            break;
        case UnsupportedAccessChainCapability::MultipleDynamicIndexes:
            SetFinalIndexRange(first_index_id, 0, 2);
            SetFinalIndexRange(final_index_id, 0, 2);
            index_ids = { first_index_id, final_index_id };
            break;
        case UnsupportedAccessChainCapability::BaseHasAddressRange:
            SetFinalIndexRange(final_index_id, 0, 2);
            SetAddressRange(base_id, 0x1000, 0x1010, 8, 8);
            index_ids = { final_index_id };
            break;
        case UnsupportedAccessChainCapability::BaseIsThreadSpecific:
            SetFinalIndexRange(final_index_id, 0, 2);
            value_meta_.resize(std::max<size_t>(value_meta_.size(), base_id + 1));
            value_meta_[base_id].flags = SPS_FLAG_THREAD_SPECIFIC;
            index_ids = { final_index_id };
            break;
        case UnsupportedAccessChainCapability::UniformDescriptorArray:
            storage_class = spv::StorageClassUniform;
            SetFinalIndexRange(final_index_id, 0, 2);
            index_ids = { final_index_id };
            break;
        case UnsupportedAccessChainCapability::UniformConstantDescriptorArray:
            storage_class = spv::StorageClassUniformConstant;
            SetFinalIndexRange(final_index_id, 0, 2);
            index_ids = { final_index_id };
            break;
        case UnsupportedAccessChainCapability::StorageBufferDescriptorArray:
            storage_class = spv::StorageClassStorageBuffer;
            SetFinalIndexRange(final_index_id, 0, 2);
            index_ids = { final_index_id };
            break;
    }

    const ::SPIRVSimulator::Type result_type =
        ::SPIRVSimulator::Type::Pointer(storage_class, CommonTypes::u64);
    const ::SPIRVSimulator::Type container_type =
        ::SPIRVSimulator::Type::RuntimeArray(CommonTypes::u64);
    EXPECT_CALL(*this, GetTypeByTypeId(result_type_id)).WillRepeatedly(ReturnRef(result_type));

    const bool is_descriptor_array =
        test_case.capability == UnsupportedAccessChainCapability::UniformDescriptorArray ||
        test_case.capability == UnsupportedAccessChainCapability::UniformConstantDescriptorArray ||
        test_case.capability == UnsupportedAccessChainCapability::StorageBufferDescriptorArray;
    if (is_descriptor_array)
    {
        EXPECT_CALL(*this, GetTargetPointerType(_)).WillOnce(Return(container_type_id));
        EXPECT_CALL(*this, GetTypeByTypeId(container_type_id))
            .WillRepeatedly(ReturnRef(container_type));
    }
    else
    {
        EXPECT_CALL(*this, GetTargetPointerType(_)).Times(0);
    }
    EXPECT_CALL(*this, GetBitsizeOfType(_)).Times(0);
    EXPECT_CALL(*this, ResolvePointerV(_)).Times(0);

    std::vector<uint32_t> words = { 0, result_type_id, result_id, base_id };
    words.insert(words.end(), index_ids.begin(), index_ids.end());
    const ::SPIRVSimulator::Instruction instruction{
        .opcode = spv::Op::OpAccessChain,
        .word_count = static_cast<uint16_t>(words.size()),
        .words = words,
    };
    const ::SPIRVSimulator::PointerV result_pointer{
        .storage_class = storage_class,
        .idx_path = std::vector<uint32_t>(index_ids.size(), 0),
    };

    EXPECT_FALSE(TrySetAddressRange(instruction, result_pointer));
    const ::SPIRVSimulator::ValueMetadata& metadata = GetMetadata(result_id);
    EXPECT_FALSE(metadata.address_range_valid);
    EXPECT_EQ(metadata.address_range_min, 0u);
    EXPECT_EQ(metadata.address_range_max, 0u);
    EXPECT_EQ(metadata.address_range_stride, 1u);
    EXPECT_EQ(metadata.address_element_size, 0u);
    EXPECT_NE(metadata.flags & SPS_FLAG_THREAD_SPECIFIC, 0u);
}

INSTANTIATE_TEST_SUITE_P(
    UnsupportedAccessChainCases,
    UnsupportedAccessChainAddressRangeCapabilityTests,
    Values(UnsupportedAccessChainCapabilityCase{
               "NonFinalDynamicIndex",
               UnsupportedAccessChainCapability::NonFinalDynamicIndex },
           UnsupportedAccessChainCapabilityCase{
               "MultipleDynamicIndexes",
               UnsupportedAccessChainCapability::MultipleDynamicIndexes },
           UnsupportedAccessChainCapabilityCase{
               "BaseHasAddressRange",
               UnsupportedAccessChainCapability::BaseHasAddressRange },
           UnsupportedAccessChainCapabilityCase{
               "BaseIsThreadSpecific",
               UnsupportedAccessChainCapability::BaseIsThreadSpecific },
           UnsupportedAccessChainCapabilityCase{
               "UniformDescriptorArray",
               UnsupportedAccessChainCapability::UniformDescriptorArray },
           UnsupportedAccessChainCapabilityCase{
               "UniformConstantDescriptorArray",
               UnsupportedAccessChainCapability::UniformConstantDescriptorArray },
           UnsupportedAccessChainCapabilityCase{
               "StorageBufferDescriptorArray",
               UnsupportedAccessChainCapability::StorageBufferDescriptorArray }),
    [](const TestParamInfo<UnsupportedAccessChainCapabilityCase>& info) {
        return std::string(info.param.name);
    });

enum class UnsupportedResultPointeeKind
{
    Bool,
    Vector,
    Struct,
    Matrix,
    FixedArray,
};

struct UnsupportedResultPointeeCase
{
    const char*                      name;
    UnsupportedResultPointeeKind     kind;
};

// Verifies the current result-pointee capability boundary while keeping the
// final dynamic index, container, storage class, and dense input range valid.
class UnsupportedResultPointeeAddressRangeTests
    : public AccessChainAddressRangeTests,
      public WithParamInterface<UnsupportedResultPointeeCase>
{};

TEST_P(UnsupportedResultPointeeAddressRangeTests, RejectsAddressRange)
{
    constexpr uint32_t result_type_id           = 5100;
    constexpr uint32_t result_id                = 5101;
    constexpr uint32_t base_id                  = 5102;
    constexpr uint32_t final_index_id           = 5103;
    constexpr uint32_t container_type_id        = 5104;
    constexpr uint32_t result_pointee_type_id   = 5105;
    constexpr uint32_t array_length_id           = 5106;

    const UnsupportedResultPointeeCase& test_case = GetParam();

    ::SPIRVSimulator::Type result_pointee_type;
    switch (test_case.kind)
    {
        case UnsupportedResultPointeeKind::Bool:
            result_pointee_type = ::SPIRVSimulator::Type::BoolT();
            break;
        case UnsupportedResultPointeeKind::Vector:
            result_pointee_type = ::SPIRVSimulator::Type::Vector(CommonTypes::u32, 2);
            break;
        case UnsupportedResultPointeeKind::Struct:
            result_pointee_type = ::SPIRVSimulator::Type::Struct(result_pointee_type_id);
            break;
        case UnsupportedResultPointeeKind::Matrix:
            result_pointee_type = ::SPIRVSimulator::Type::Matrix(CommonTypes::uvec2, 2);
            break;
        case UnsupportedResultPointeeKind::FixedArray:
            result_pointee_type =
                ::SPIRVSimulator::Type::Array(CommonTypes::u32, array_length_id);
            break;
    }

    const ::SPIRVSimulator::Type result_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, result_pointee_type_id);
    const ::SPIRVSimulator::Type container_type =
        ::SPIRVSimulator::Type::RuntimeArray(result_pointee_type_id);

    EXPECT_CALL(*this, GetTypeByTypeId(result_type_id)).WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByTypeId(container_type_id)).WillRepeatedly(ReturnRef(container_type));
    EXPECT_CALL(*this, GetTypeByTypeId(result_pointee_type_id))
        .WillRepeatedly(ReturnRef(result_pointee_type));
    EXPECT_CALL(*this, GetTargetPointerType(_)).WillOnce(Return(container_type_id));
    EXPECT_CALL(*this, GetBitsizeOfType(_)).Times(0);
    EXPECT_CALL(*this, ResolvePointerV(_)).Times(0);

    SetFinalIndexRange(final_index_id, 0, 2);
    std::vector<uint32_t> words = { 0, result_type_id, result_id, base_id, final_index_id };
    const ::SPIRVSimulator::Instruction instruction{
        .opcode = spv::Op::OpAccessChain,
        .word_count = static_cast<uint16_t>(words.size()),
        .words = words,
    };
    const ::SPIRVSimulator::PointerV result_pointer{
        .storage_class = spv::StorageClassPhysicalStorageBuffer,
        .idx_path = { 0 },
    };

    EXPECT_FALSE(TrySetAddressRange(instruction, result_pointer));
    EXPECT_FALSE(GetMetadata(result_id).address_range_valid);
    EXPECT_NE(GetMetadata(result_id).flags & SPS_FLAG_THREAD_SPECIFIC, 0u);
}

INSTANTIATE_TEST_SUITE_P(
    UnsupportedResultPointeeCases,
    UnsupportedResultPointeeAddressRangeTests,
    Values(UnsupportedResultPointeeCase{ "BoolResult", UnsupportedResultPointeeKind::Bool },
           UnsupportedResultPointeeCase{ "VectorResult", UnsupportedResultPointeeKind::Vector },
           UnsupportedResultPointeeCase{ "StructResult", UnsupportedResultPointeeKind::Struct },
           UnsupportedResultPointeeCase{ "MatrixResult", UnsupportedResultPointeeKind::Matrix },
           UnsupportedResultPointeeCase{ "FixedArrayResult",
                                         UnsupportedResultPointeeKind::FixedArray }),
    [](const TestParamInfo<UnsupportedResultPointeeCase>& info) {
        return std::string(info.param.name);
    });

struct UnsupportedDenseFinalLayoutCase
{
    const char* name;
    uint64_t    index_min;
    uint64_t    index_max;
    uint64_t    index_stride;
    uint64_t    container_byte_step;
};

// Verifies that multi-element uint64 ranges with padding or skipped indexes are
// rejected only after their resolved byte stride proves the layout is non-dense.
class UnsupportedDenseFinalLayoutTests
    : public AccessChainAddressRangeTests,
      public WithParamInterface<UnsupportedDenseFinalLayoutCase>
{};

TEST_P(UnsupportedDenseFinalLayoutTests, RejectsAddressRange)
{
    constexpr uint32_t result_type_id    = 5300;
    constexpr uint32_t result_id         = 5301;
    constexpr uint32_t base_id           = 5302;
    constexpr uint32_t final_index_id    = 5303;
    constexpr uint32_t container_type_id = 5304;

    const UnsupportedDenseFinalLayoutCase& test_case = GetParam();
    const ::SPIRVSimulator::Type result_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClassPhysicalStorageBuffer, CommonTypes::u64);
    const ::SPIRVSimulator::Type container_type =
        ::SPIRVSimulator::Type::RuntimeArray(CommonTypes::u64);

    EXPECT_CALL(*this, GetTypeByTypeId(result_type_id)).WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByTypeId(container_type_id)).WillRepeatedly(ReturnRef(container_type));
    EXPECT_CALL(*this, GetTargetPointerType(_)).WillOnce(Return(container_type_id));
    EXPECT_CALL(*this, GetBitsizeOfType(CommonTypes::u64)).WillOnce(Return(64));
    EXPECT_CALL(*this, ResolvePointerV(_))
        .Times(3)
        .WillRepeatedly([this, &test_case](const ::SPIRVSimulator::PointerV& pointer) {
            return std::pair<std::byte*, uint64_t>{
                backing_.data(), pointer.idx_path.back() * test_case.container_byte_step
            };
        });

    SetFinalIndexRange(final_index_id,
                       test_case.index_min,
                       test_case.index_max,
                       test_case.index_stride);
    std::vector<uint32_t> words = { 0, result_type_id, result_id, base_id, final_index_id };
    const ::SPIRVSimulator::Instruction instruction{
        .opcode = spv::Op::OpAccessChain,
        .word_count = static_cast<uint16_t>(words.size()),
        .words = words,
    };
    const ::SPIRVSimulator::PointerV result_pointer{
        .storage_class = spv::StorageClassPhysicalStorageBuffer,
        .idx_path = { 0 },
    };

    EXPECT_FALSE(TrySetAddressRange(instruction, result_pointer));
    const ::SPIRVSimulator::ValueMetadata& metadata = GetMetadata(result_id);
    EXPECT_FALSE(metadata.address_range_valid);
    EXPECT_EQ(metadata.address_range_min, 0u);
    EXPECT_EQ(metadata.address_range_max, 0u);
    EXPECT_EQ(metadata.address_range_stride, 1u);
    EXPECT_EQ(metadata.address_element_size, 0u);
    EXPECT_NE(metadata.flags & SPS_FLAG_THREAD_SPECIFIC, 0u);
}

INSTANTIATE_TEST_SUITE_P(
    NonDenseFinalLayoutCases,
    UnsupportedDenseFinalLayoutTests,
    Values(UnsupportedDenseFinalLayoutCase{ "PaddedScalarElements", 0, 2, 1, 16 },
           UnsupportedDenseFinalLayoutCase{ "SparseSelectedIndexes", 0, 4, 2, 8 }),
    [](const TestParamInfo<UnsupportedDenseFinalLayoutCase>& info) {
        return std::string(info.param.name);
    });

// Verify failure handling while dense AccessChain ranges are consumed to collect
// nested PhysicalStorageBuffer pointers from read-side pointer tables.
class AccessChainReadSidePointerTests : public SPIRVSimulatorMockBase, public Test
{
  public:
    MOCK_METHOD(std::optional<::SPIRVSimulator::Value>,
                ReadPointer,
                (const ::SPIRVSimulator::PointerV& pointer),
                (override));

  protected:
    void SetUp() override
    {
        memory_flag_tracker_ = nullptr;
        simulation_results_  = &simulation_results_storage_;
        verbose_             = false;
    }

    bool QueueDenseReadSidePointers(const ::SPIRVSimulator::Instruction& instruction,
                                    uint32_t result_id,
                                    const ::SPIRVSimulator::PointerV& representative_pointer)
    {
        return TryQueueDenseReadSidePbufferPointers(
            instruction, result_id, representative_pointer);
    }

    void ExecuteAccessChain(const ::SPIRVSimulator::Instruction& instruction)
    {
        Op_AccessChain(instruction);
    }

    size_t QueuedPointerPairCount() const
    {
        return pointers_to_physical_address_pointers_.size();
    }

    bool ReadSidePointerCoverageIncomplete() const
    {
        return simulation_results_storage_.read_side_pointer_coverage_incomplete;
    }

    void SetDenseQueueMetadata(uint32_t result_id, uint32_t final_index_id)
    {
        value_meta_.resize(std::max(result_id, final_index_id) + 1);

        ::SPIRVSimulator::ValueMetadata& result_meta = value_meta_[result_id];
        result_meta.address_range_valid               = true;
        result_meta.address_range_min                 = 100;
        result_meta.address_range_max                 = 116;
        result_meta.address_range_stride              = 8;
        result_meta.address_element_size              = 8;

        value_meta_[final_index_id].value_range =
            ::SPIRVSimulator::DenseValueRangeInfo{ true, true, true, 0, 2, 1 };
    }

    static ::SPIRVSimulator::PointerV MakePointer(uint32_t index)
    {
        return ::SPIRVSimulator::PointerV{
            1,
            0,
            1,
            1,
            spv::StorageClass::StorageClassFunction,
            { index },
        };
    }

    ::SPIRVSimulator::SimulationResults simulation_results_storage_{};
};

TEST_F(AccessChainReadSidePointerTests, DenseQueueReadFailureDoesNotCommitPartialPairs)
{
    constexpr uint32_t result_id      = 1000;
    constexpr uint32_t final_index_id = 1001;

    SetDenseQueueMetadata(result_id, final_index_id);

    std::vector<uint32_t> words = { 0, 0, result_id, 0, final_index_id };
    const ::SPIRVSimulator::Instruction instruction{
        .opcode = spv::Op::OpAccessChain,
        .word_count = static_cast<uint16_t>(words.size()),
        .words = words,
    };
    const ::SPIRVSimulator::PointerV representative_pointer = MakePointer(0);
    const ::SPIRVSimulator::PointerV loaded_pointer         = MakePointer(7);

    InSequence sequence;
    EXPECT_CALL(*this, ReadPointer(_))
        .WillOnce([loaded_pointer](const ::SPIRVSimulator::PointerV& pointer)
                      -> std::optional<::SPIRVSimulator::Value> {
            EXPECT_EQ(pointer.idx_path, std::vector<uint32_t>{ 0 });
            return ::SPIRVSimulator::Value{ loaded_pointer };
        });
    EXPECT_CALL(*this, ReadPointer(_))
        .WillOnce([](const ::SPIRVSimulator::PointerV& pointer)
                      -> std::optional<::SPIRVSimulator::Value> {
            EXPECT_EQ(pointer.idx_path, std::vector<uint32_t>{ 1 });
            return std::nullopt;
        });

    EXPECT_FALSE(QueueDenseReadSidePointers(instruction, result_id, representative_pointer));
    EXPECT_EQ(QueuedPointerPairCount(), 0u);
}

TEST_F(AccessChainReadSidePointerTests, ThreadSpecificFallbackReadFailureReportsIncompleteCoverage)
{
    constexpr uint32_t base_type_id            = 1100;
    constexpr uint32_t result_type_id          = 1101;
    constexpr uint32_t nested_pointer_type_id  = 1102;
    constexpr uint32_t base_id                 = 1103;
    constexpr uint32_t result_id               = 1104;
    constexpr uint32_t final_index_id          = 1105;

    const ::SPIRVSimulator::Type base_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClass::StorageClassFunction, CommonTypes::u64);
    const ::SPIRVSimulator::Type result_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClass::StorageClassFunction, nested_pointer_type_id);
    const ::SPIRVSimulator::Type nested_pointer_type = ::SPIRVSimulator::Type::Pointer(
        spv::StorageClass::StorageClassPhysicalStorageBuffer, CommonTypes::u64);

    ::SPIRVSimulator::Value base_value = ::SPIRVSimulator::PointerV{
        1,
        0,
        base_type_id,
        base_id,
        spv::StorageClass::StorageClassFunction,
        {},
    };
    ::SPIRVSimulator::Value final_index_value = uint64_t{ 0 };

    value_meta_.resize(final_index_id + 1);
    value_meta_[final_index_id].flags |= SPS_FLAG_THREAD_SPECIFIC;

    EXPECT_CALL(*this, GetValue(base_id)).WillOnce(ReturnRef(base_value));
    EXPECT_CALL(*this, GetValue(final_index_id)).WillOnce(ReturnRef(final_index_value));
    EXPECT_CALL(*this, GetTypeByResultId(base_id)).WillOnce(ReturnRef(base_type));
    EXPECT_CALL(*this, GetTypeByTypeId(result_type_id)).WillRepeatedly(ReturnRef(result_type));
    EXPECT_CALL(*this, GetTypeByTypeId(nested_pointer_type_id))
        .WillOnce(ReturnRef(nested_pointer_type));
    EXPECT_CALL(*this, SetValue(result_id, _, true)).Times(1);
    EXPECT_CALL(*this, TransferFlags(result_id, TypedEq<uint32_t>(base_id))).Times(1);
    EXPECT_CALL(*this, ReadPointer(_)).WillOnce(Return(std::nullopt));

    std::vector<uint32_t> words = { 0, result_type_id, result_id, base_id, final_index_id };
    const ::SPIRVSimulator::Instruction instruction{
        .opcode = spv::Op::OpAccessChain,
        .word_count = static_cast<uint16_t>(words.size()),
        .words = words,
    };

    ExecuteAccessChain(instruction);

    EXPECT_TRUE(ReadSidePointerCoverageIncomplete());
    EXPECT_EQ(QueuedPointerPairCount(), 0u);
}
