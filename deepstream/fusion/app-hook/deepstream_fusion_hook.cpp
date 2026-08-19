#include "deepstream_fusion_hook.hpp"

#include <array>
#include <atomic>
#include <cstring>
#include <new>

#include "deepsafe_fusion/c_api.h"

namespace {

constexpr guint kMaximumSources = 12U;

void set_error(char* buffer, gsize size, const char* message) {
  if (buffer == nullptr || size == 0U) {
    return;
  }
  g_strlcpy(buffer, message == nullptr ? "fusion hook failure" : message, size);
}

struct HookState {
  DeepsafeFusionHandle* fusion{nullptr};
  GstElement* metamux{nullptr};
  GstPad* source_pad{nullptr};
  gulong probe_id{0U};
  std::atomic_bool error_posted{false};
};

GstPadProbeReturn fusion_probe(GstPad*, GstPadProbeInfo* info,
                               gpointer user_data) {
  auto* state = static_cast<HookState*>(user_data);
  auto* buffer = info == nullptr ? nullptr : GST_PAD_PROBE_INFO_BUFFER(info);
  std::array<char, 512> error{};
  if (state == nullptr || buffer == nullptr ||
      deepsafe_fusion_process_gst_buffer_v1(
          state->fusion, buffer, error.data(), error.size()) < 0) {
    if (state != nullptr && state->metamux != nullptr &&
        !state->error_posted.exchange(true)) {
      GST_ELEMENT_ERROR(state->metamux, STREAM, FAILED,
                        ("DeepSafe canonical fusion rejected a batch"),
                        ("%s", error[0] == '\0' ? "invalid probe buffer" :
                                                 error.data()));
    }
    return GST_PAD_PROBE_DROP;
  }
  return GST_PAD_PROBE_OK;
}

}  // namespace

struct DeepsafeFusionAppHook {
  HookState state{};
};

DeepsafeFusionAppHook* deepsafe_fusion_app_hook_create(
    char* error_buffer, gsize error_buffer_size) {
  if (deepsafe_fusion_abi_version_v1() != DEEPSAFE_FUSION_ABI_VERSION_V1) {
    set_error(error_buffer, error_buffer_size,
              "DeepSafe fusion C ABI version mismatch");
    return nullptr;
  }
  auto* hook = new (std::nothrow) DeepsafeFusionAppHook();
  if (hook == nullptr) {
    set_error(error_buffer, error_buffer_size,
              "cannot allocate DeepSafe fusion hook");
    return nullptr;
  }
  hook->state.fusion = deepsafe_fusion_create_from_env_v1(
      error_buffer, static_cast<size_t>(error_buffer_size));
  if (hook->state.fusion == nullptr) {
    delete hook;
    return nullptr;
  }
  return hook;
}

gboolean deepsafe_fusion_app_hook_install(
    DeepsafeFusionAppHook* hook, GstElement* metamux, guint source_count,
    char* error_buffer, gsize error_buffer_size) {
  if (hook == nullptr || hook->state.fusion == nullptr || metamux == nullptr) {
    set_error(error_buffer, error_buffer_size,
              "fusion hook/metamux is null");
    return FALSE;
  }
  if (hook->state.probe_id != 0U || hook->state.source_pad != nullptr) {
    set_error(error_buffer, error_buffer_size,
              "fusion hook was already installed");
    return FALSE;
  }
  if (source_count == 0U || source_count > kMaximumSources) {
    set_error(error_buffer, error_buffer_size,
              "fusion deployment source count must be in [1,12]");
    return FALSE;
  }
  auto* pad = gst_element_get_static_pad(metamux, "src");
  if (pad == nullptr) {
    set_error(error_buffer, error_buffer_size,
              "cannot acquire nvdsmetamux src pad");
    return FALSE;
  }
  hook->state.metamux = GST_ELEMENT(gst_object_ref(metamux));
  hook->state.source_pad = pad;
  hook->state.probe_id = gst_pad_add_probe(
      pad, GST_PAD_PROBE_TYPE_BUFFER, fusion_probe, &hook->state, nullptr);
  if (hook->state.probe_id == 0U) {
    gst_object_unref(hook->state.source_pad);
    gst_object_unref(hook->state.metamux);
    hook->state.source_pad = nullptr;
    hook->state.metamux = nullptr;
    set_error(error_buffer, error_buffer_size,
              "cannot install canonical fusion pad probe");
    return FALSE;
  }
  return TRUE;
}

void deepsafe_fusion_app_hook_destroy(DeepsafeFusionAppHook* hook) {
  if (hook == nullptr) {
    return;
  }
  if (hook->state.source_pad != nullptr && hook->state.probe_id != 0U) {
    gst_pad_remove_probe(hook->state.source_pad, hook->state.probe_id);
  }
  hook->state.probe_id = 0U;
  if (hook->state.source_pad != nullptr) {
    gst_object_unref(hook->state.source_pad);
  }
  if (hook->state.metamux != nullptr) {
    gst_object_unref(hook->state.metamux);
  }
  deepsafe_fusion_destroy_v1(hook->state.fusion);
  hook->state.source_pad = nullptr;
  hook->state.metamux = nullptr;
  hook->state.fusion = nullptr;
  delete hook;
}
