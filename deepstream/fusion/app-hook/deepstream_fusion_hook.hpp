#pragma once

#include <gst/gst.h>

struct DeepsafeFusionAppHook;

// Configuration is read exclusively through DEEPSAFE_FUSION_CONFIG and
// DEEPSAFE_FUSION_CONFIG_SHA256.  A missing/mismatched value is fatal.
DeepsafeFusionAppHook* deepsafe_fusion_app_hook_create(
    char* error_buffer, gsize error_buffer_size);

// Installs the canonical fusion probe after nvdsmetamux.  The runtime accepts
// partial batches but the configured deployment may never exceed 12 sources.
gboolean deepsafe_fusion_app_hook_install(
    DeepsafeFusionAppHook* hook, GstElement* metamux, guint source_count,
    char* error_buffer, gsize error_buffer_size);

// Removes the probe before releasing its state and references.
void deepsafe_fusion_app_hook_destroy(DeepsafeFusionAppHook* hook);
