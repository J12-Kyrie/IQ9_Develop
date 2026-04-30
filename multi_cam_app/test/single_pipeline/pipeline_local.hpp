#ifndef SINGLE_PIPELINE_PIPELINE_LOCAL_HPP
#define SINGLE_PIPELINE_PIPELINE_LOCAL_HPP

#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

#include <gst/gst.h>

namespace single_pipeline {

std::string MakeElementName(const std::string& base, uint32_t channel_id);

GstElement* MakeElement(const char* factory_name,
                        const std::string& element_name,
                        std::string* out_error);

bool AddElements(GstBin* bin,
                 const std::initializer_list<GstElement*>& elements,
                 std::string* out_error);

bool AddElements(GstBin* bin,
                 const std::vector<GstElement*>& elements,
                 std::string* out_error);

bool LinkElements(const std::vector<GstElement*>& elements,
                  std::string* out_error);

bool SetEnumPropertyByNick(GstElement* element,
                           const char* property_name,
                           const std::string& value_nick,
                           std::string* out_error);

}  // namespace single_pipeline

#endif  // SINGLE_PIPELINE_PIPELINE_LOCAL_HPP
