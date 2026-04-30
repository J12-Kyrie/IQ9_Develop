#include "pipeline_local.hpp"

#include <sstream>

namespace single_pipeline {

std::string MakeElementName(const std::string& base, uint32_t channel_id) {
  return base + "_ch" + std::to_string(channel_id);
}

GstElement* MakeElement(const char* factory_name,
                        const std::string& element_name,
                        std::string* out_error) {
  GstElement* element = gst_element_factory_make(factory_name, element_name.c_str());
  if (element == nullptr && out_error != nullptr) {
    *out_error = "Failed to create element '" + std::string(factory_name) + "' as '" +
                 element_name + "'";
  }
  return element;
}

bool AddElements(GstBin* bin,
                 const std::initializer_list<GstElement*>& elements,
                 std::string* out_error) {
  return AddElements(bin, std::vector<GstElement*>(elements), out_error);
}

bool AddElements(GstBin* bin,
                 const std::vector<GstElement*>& elements,
                 std::string* out_error) {
  if (bin == nullptr) {
    if (out_error != nullptr) {
      *out_error = "AddElements received null bin";
    }
    return false;
  }

  for (GstElement* element : elements) {
    if (element == nullptr) {
      if (out_error != nullptr) {
        *out_error = "AddElements received null element";
      }
      return false;
    }
    if (!gst_bin_add(bin, element)) {
      if (out_error != nullptr) {
        *out_error = "Failed to add element to pipeline bin";
      }
      return false;
    }
  }

  return true;
}

bool LinkElements(const std::vector<GstElement*>& elements,
                  std::string* out_error) {
  if (elements.size() < 2U) {
    return true;
  }
  for (size_t i = 0U; i + 1U < elements.size(); ++i) {
    GstElement* src = elements[i];
    GstElement* sink = elements[i + 1U];
    if ((src == nullptr) || (sink == nullptr)) {
      if (out_error != nullptr) {
        *out_error = "LinkElements received null element";
      }
      return false;
    }
    if (!gst_element_link(src, sink)) {
      if (out_error != nullptr) {
        std::ostringstream oss;
        oss << "Failed to link element index " << i << " to " << (i + 1U);
        *out_error = oss.str();
      }
      return false;
    }
  }
  return true;
}

bool SetEnumPropertyByNick(GstElement* element,
                           const char* property_name,
                           const std::string& value_nick,
                           std::string* out_error) {
  if ((element == nullptr) || (property_name == nullptr)) {
    if (out_error != nullptr) {
      *out_error = "SetEnumPropertyByNick received null argument";
    }
    return false;
  }

  GObjectClass* klass = G_OBJECT_GET_CLASS(element);
  GParamSpec* param = g_object_class_find_property(klass, property_name);
  if (param == nullptr) {
    if (out_error != nullptr) {
      *out_error = "Enum property not found: " + std::string(property_name);
    }
    return false;
  }
  if (!G_IS_PARAM_SPEC_ENUM(param)) {
    if (out_error != nullptr) {
      *out_error = "Property is not enum: " + std::string(property_name);
    }
    return false;
  }

  auto* enum_spec = G_PARAM_SPEC_ENUM(param);
  GEnumClass* enum_class = G_ENUM_CLASS(g_type_class_ref(param->value_type));
  if (enum_class == nullptr) {
    if (out_error != nullptr) {
      *out_error = "Failed to load enum class for property: " + std::string(property_name);
    }
    return false;
  }

  const GEnumValue* enum_value = g_enum_get_value_by_nick(enum_class, value_nick.c_str());
  if (enum_value == nullptr) {
    g_type_class_unref(enum_class);
    if (out_error != nullptr) {
      *out_error = "Enum nick not found for " + std::string(property_name) + ": " + value_nick;
    }
    return false;
  }

  g_object_set(G_OBJECT(element), property_name, enum_value->value, nullptr);
  g_type_class_unref(enum_class);

  (void)enum_spec;
  return true;
}

}  // namespace single_pipeline
