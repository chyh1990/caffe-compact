CROSS_COMPILE?=
CXX=$(CROSS_COMPILE)g++
AR=$(CROSS_COMPILE)ar
PROJECT := caffe
STATIC_NAME := lib$(PROJECT).a
USE_EIGEN?=y

CXX_SRCS := $(shell find src/$(PROJECT) ! -name "test_*.cpp" -name "*.cpp")
HXX_SRCS := $(shell find include/$(PROJECT) ! -name "*.hpp")
PROTO_SRCS := $(wildcard src/$(PROJECT)/proto/*.proto)

PROTO_GEN_HEADER := ${PROTO_SRCS:.proto=.pb.h}
PROTO_GEN_CC := ${PROTO_SRCS:.proto=.pb.cc}

BUILD_DIR := build
CXX_OBJS := $(addprefix $(BUILD_DIR)/, ${CXX_SRCS:.cpp=.o})
PROTO_OBJS := $(addprefix $(BUILD_DIR)/, ${PROTO_GEN_CC:.cc=.o})
OBJS := $(PROTO_OBJS) $(CXX_OBJS)

INCLUDE_DIRS += ./src ./include ./protobuf-2.4.1/build/include
CXXFLAGS+=-std=gnu++0x
CXXFLAGS+=$(EXTRA_CXXFLAGS)
CXXFLAGS+=-fvisibility=hidden #hide symbols for static lib
LDFLAGS+=-L./protobuf-2.4.1/build/lib
LIBRARIES:=protobuf pthread 

ifeq ($(USE_EIGEN), y)
	CXXFLAGS += -DUSE_EIGEN
	CXXFLAGS += -I./eigen3
else
	LIBRARIES += cblas
endif

CXXFLAGS += -DCPU_ONLY

COMMON_FLAGS := -O2 $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
CXXFLAGS +=  -fPIC $(COMMON_FLAGS)
LDFLAGS += $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir)) \
		$(foreach library,$(LIBRARIES),-l$(library))

all: init $(STATIC_NAME) 

init:
	@ mkdir -p $(foreach obj,$(OBJS),$(dir $(obj)))

$(OBJS): $(PROTO_GEN_CC) $(HXX_SRCS)

$(BUILD_DIR)/src/$(PROJECT)/%.o: src/$(PROJECT)/%.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

$(BUILD_DIR)/src/$(PROJECT)/layers/%.o: src/$(PROJECT)/layers/%.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

$(BUILD_DIR)/src/$(PROJECT)/proto/%.o: src/$(PROJECT)/proto/%.cc
	$(CXX) $< $(CXXFLAGS) -c -o $@

$(PROTO_GEN_CC): $(PROTO_SRCS)
	protoc --proto_path=src --cpp_out=src $(PROTO_SRCS)
	mkdir -p include/$(PROJECT)/proto
	cp $(PROTO_GEN_HEADER) include/$(PROJECT)/proto/
	@echo

$(STATIC_NAME): init $(PROTO_OBJS) $(OBJS)
	$(AR) rcs $(STATIC_NAME) $(PROTO_OBJS) $(OBJS)
	@echo

feat_net_raw: feat_net_raw.cpp $(STATIC_NAME)
	$(CXX) $< $(CXXFLAGS) -o $@ -L. -lcaffe $(LDFLAGS) -lpthread

align_test: align_test.cpp $(STATIC_NAME)
	$(CXX) $< $(CXXFLAGS) -o $@ -L. -lcaffe $(LDFLAGS) $(shell pkg-config --libs opencv)

clean:
	@- $(RM) $(NAME) $(STATIC_NAME)
	@- $(RM) $(PROTO_GEN_HEADER) $(PROTO_GEN_CC) $(PROTO_GEN_PY)
	@- $(RM) include/$(PROJECT)/proto/$(PROJECT).pb.h
	@- $(RM) -rf $(BUILD_DIR)
	@- rm -f feat_net_raw


