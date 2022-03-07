// Copyright 2019-2021 Varjo Technologies Oy. All rights reserved.

#include "AppLogic.hpp"

#include <cstdio>
#include <vector>
#include <string>
#include <fstream>
#include <set>
#include <thread>

#include <Varjo.h>
#include <Varjo_events.h>
#include <Varjo_mr.h>
#include <Varjo_d3d11.h>
#include <glm/gtx/matrix_decompose.hpp>

#include "D3D11MultiLayerView.hpp"

// VarjoExamples namespace contains simple example wrappers for using Varjo API features.
// These are only meant to be used in SDK example applications. In your own application,
// use your own production quality integration layer.
using namespace VarjoExamples;

//---------------------------------------------------------------------------

AppLogic::~AppLogic()
{
    // Free camera manager resources
    m_camera.reset();

    // Free data stremer resources
    m_streamer.reset();

    // Free scene resources
    m_scene.reset();

    // Free view resources
    m_varjoView.reset();

    // Free renderer resources
    m_renderer.reset();

    // Shutdown the varjo session. Can't check errors anymore after this.
    LOG_DEBUG("Shutting down Varjo session..");
    varjo_SessionShutDown(m_session);
    m_session = nullptr;
}

bool AppLogic::init(GfxContext& context)
{
    // Initialize the varjo session.
    LOG_DEBUG("Initializing Varjo session..");
    m_session = varjo_SessionInit();
    if (CHECK_VARJO_ERR(m_session) != varjo_NoError) {
        LOG_ERROR("Creting Varjo session failed.");
        return false;
    }

    {
        // Get graphics adapter used by Varjo session
        auto dxgiAdapter = D3D11MultiLayerView::getAdapter(m_session);

        // Init graphics
        context.init(dxgiAdapter.Get());

        // Create D3D11 renderer instance.
        auto d3d11Renderer = std::make_unique<D3D11Renderer>(dxgiAdapter.Get());

        // Create varjo multi layer view
        m_varjoView = std::make_unique<D3D11MultiLayerView>(m_session, *d3d11Renderer);

        // Store as generic renderer
        m_renderer = std::move(d3d11Renderer);
    }

    // Create scene instance
    m_scene = std::make_unique<MRScene>(*m_renderer);

    // Create data streamer instance
    m_streamer = std::make_unique<DataStreamer>(m_session);

    // Create mixed reality camera manager instance
    m_camera = std::make_unique<CameraManager>(m_session);

    // Check if Mixed Reality features are available.
    varjo_Bool mixedRealityAvailable = varjo_False;
    varjo_SyncProperties(m_session);
    CHECK_VARJO_ERR(m_session);
    if (varjo_HasProperty(m_session, varjo_PropertyKey_MRAvailable)) {
        mixedRealityAvailable = varjo_GetPropertyBool(m_session, varjo_PropertyKey_MRAvailable);
    }

    // Handle mixed reality availability
    onMixedRealityAvailable(mixedRealityAvailable == varjo_True, false);

    if (mixedRealityAvailable == varjo_True) {
        LOG_INFO("Varjo Mixed Reality features available!");

        // Optionally reset camera at startup
        constexpr bool resetCameraAtStart = true;
        if (resetCameraAtStart) {
            // Reset camera properties to defaults
            m_camera->resetPropertiesToDefaults();

            // Set camera exposure and white balance to auto
            m_camera->setAutoMode(varjo_CameraPropertyType_ExposureTime);
            m_camera->setAutoMode(varjo_CameraPropertyType_WhiteBalance);
        }

    } else {
        LOG_WARNING("Varjo Mixed Reality features not available!");
    }

    return true;
}

void AppLogic::setVideoRendering(bool enabled)
{
    varjo_MRSetVideoRender(m_session, enabled ? varjo_True : varjo_False);
    if (CHECK_VARJO_ERR(m_session) == varjo_NoError) {
        LOG_INFO("Video rendering: %s", enabled ? "ON" : "OFF");
    }
    m_appState.options.videoRenderingEnabled = enabled;
}

void AppLogic::setState(const AppState& appState, bool force)
{
    // Store previous state and set new one
    const auto prevState = m_appState;
    m_appState = appState;

    // Client priority
    if (force || appState.options.clientPriority != prevState.options.clientPriority) {
        varjo_SessionSetPriority(m_session, appState.options.clientPriority);
        if (CHECK_VARJO_ERR(m_session) == varjo_NoError) {
            LOG_INFO("Client priority: %i", appState.options.clientPriority);
        }
    }
    // Check for mixed reality availability
    if (!m_appState.general.mrAvailable) {
        // Toggle video rendering off
        if (m_appState.options.videoRenderingEnabled) {
            setVideoRendering(false);
        }
        return;
    }

    // Video rendering
    if (force || appState.options.videoRenderingEnabled != prevState.options.videoRenderingEnabled) {
        setVideoRendering(appState.options.videoRenderingEnabled);
    }

    // Video depth estimation
    if (force || appState.options.VideoDepthEstimationEnabled != prevState.options.VideoDepthEstimationEnabled) {
        varjo_MRSetVideoDepthEstimation(m_session, appState.options.VideoDepthEstimationEnabled ? varjo_True : varjo_False);
        if (CHECK_VARJO_ERR(m_session) == varjo_NoError) {
            LOG_INFO("Video depth estimation: %s", appState.options.VideoDepthEstimationEnabled ? "ON" : "OFF");
        }
    }

    // Chroma keying
    if (force || appState.options.chromaKeyingEnabled != prevState.options.chromaKeyingEnabled) {
        varjo_MRSetChromaKey(m_session, appState.options.chromaKeyingEnabled ? varjo_True : varjo_False);
        if (CHECK_VARJO_ERR(m_session) == varjo_NoError) {
            LOG_INFO("Chroma keying: %s", appState.options.chromaKeyingEnabled ? "ON" : "OFF");
        }
    }

    // VR view offset
    if (force || appState.options.vrViewOffset != prevState.options.vrViewOffset) {
        varjo_MRSetVRViewOffset(m_session, appState.options.vrViewOffset);
        if (CHECK_VARJO_ERR(m_session) == varjo_NoError) {
            LOG_INFO("VR view offset: %.1f", appState.options.vrViewOffset);
        }
    }

    // VR rendering
    if (force || appState.options.renderVREnabled != prevState.options.renderVREnabled) {
        LOG_INFO("Rendering VR layer: %s", appState.options.renderVREnabled ? "ON" : "OFF");
    }

    // VR depth submit
    if (force || appState.options.submitVRDepthEnabled != prevState.options.submitVRDepthEnabled) {
        LOG_INFO("Submitting VR depth: %s", appState.options.submitVRDepthEnabled ? "ON" : "OFF");
    }

    // VR depth test range
    if (force || (appState.options.vrDepthTestRangeEnabled != prevState.options.vrDepthTestRangeEnabled) ||
        (appState.options.vrDepthTestRangeValue != prevState.options.vrDepthTestRangeValue)) {
        LOG_INFO("Depth test range: %s %.2f", (appState.options.vrDepthTestRangeEnabled ? "ON" : "OFF"), appState.options.vrDepthTestRangeValue);
    }

    // React MR events
    if (force || appState.options.reactToConnectionEvents != prevState.options.reactToConnectionEvents) {
        LOG_INFO("Handling connection events: %s", appState.options.reactToConnectionEvents ? "ON" : "OFF");
    }

    // VR background
    if (force || appState.options.drawVRBackgroundEnabled != prevState.options.drawVRBackgroundEnabled) {
        std::string warn = (appState.options.videoRenderingEnabled && appState.options.drawVRBackgroundEnabled) ? " (not visible with VST)" : "";
        LOG_INFO("Drawing VR background: %s %s", appState.options.drawVRBackgroundEnabled ? "ON" : "OFF", warn.c_str());
    }

    // VR ambient light
    if (force || appState.options.ambientLightTempK != prevState.options.ambientLightTempK ||
        appState.options.ambientLightGainRGB != prevState.options.ambientLightGainRGB) {
        LOG_INFO("VR ambient light (%dK): [%f, %f, %f]", appState.options.ambientLightTempK, appState.options.ambientLightGainRGB[0],
            appState.options.ambientLightGainRGB[1], appState.options.ambientLightGainRGB[2]);
    }

    // VR color correction
    if (force || appState.options.vrColorCorrectionEnabled != prevState.options.vrColorCorrectionEnabled) {
        LOG_INFO("Color correcting VR: %s", appState.options.vrColorCorrectionEnabled ? "ON" : "OFF");

        const auto reqType = varjo_StreamType_DistortedColor;
        const auto reqFormat = m_colorStreamFormat;
        const auto reqChannels = varjo_ChannelFlag_None;

        varjo_ChannelFlag runningChannels = varjo_ChannelFlag_None;
        if (appState.options.vrColorCorrectionEnabled) {
            if (m_streamer->isStreaming(reqType, reqFormat, runningChannels)) {
                LOG_INFO("Already streaming color stream..");
            } else {
                LOG_INFO("Starting metadata only color stream..");
                m_streamer->startDataStream(reqType, reqFormat, reqChannels);
            }
        } else {
            if (m_streamer->isStreaming(reqType, reqFormat, runningChannels)) {
                if (runningChannels == reqChannels) {
                    LOG_INFO("Stop metadata only color stream..");
                    m_streamer->stopDataStream(reqType, reqFormat);
                } else {
                    LOG_INFO("Keep streaming color with data..");
                }
            } else {
                // Not streaming, nothing to do.
            }
        }
    }

    // Data stream buffer handling
    if (force || appState.options.delayedBufferHandlingEnabled != prevState.options.delayedBufferHandlingEnabled) {
        m_streamer->setDelayedBufferHandlingEnabled(appState.options.delayedBufferHandlingEnabled);
        LOG_INFO("Buffer handling: %s", appState.options.delayedBufferHandlingEnabled ? "DELAYED" : "IMMEDIATE");
    }

    // Data stream: YUV
    if (force || appState.options.dataStreamColorEnabled != prevState.options.dataStreamColorEnabled) {
        const auto streamType = varjo_StreamType_DistortedColor;
        const auto streamFormat = m_colorStreamFormat;
        const auto streamChannels = varjo_ChannelFlag_Left | varjo_ChannelFlag_Right;
        varjo_ChannelFlag currentChannels = varjo_ChannelFlag_None;

        if (appState.options.dataStreamColorEnabled) {
            if (m_streamer->isStreaming(streamType, streamFormat, currentChannels)) {
                if (currentChannels == streamChannels) {
                    // If running stream channels match, match we want to stop it
                    m_streamer->stopDataStream(streamType, streamFormat);
                    if (appState.options.vrColorCorrectionEnabled) {
                        // If color correction is enabled, start stream with no data channels to
                        // keep getting camera metadata for exposure adjustment
                        LOG_INFO("Switching to metadata only color stream..");
                        m_streamer->startDataStream(streamType, streamFormat, varjo_ChannelFlag_None);
                    }
                } else {
                    // When color correction is enabled, stop stream and start new stream with data channels
                    LOG_INFO("Switching to color stream with data channels..");
                    assert(appState.options.vrColorCorrectionEnabled);
                    m_streamer->stopDataStream(streamType, streamFormat);
                    m_streamer->startDataStream(streamType, streamFormat, streamChannels);
                }
            } else {
                // No streams running, just start our color data stream
                m_streamer->startDataStream(streamType, streamFormat, streamChannels);
            }
        } else {
            if (m_streamer->isStreaming(streamType, streamFormat, currentChannels)) {
                if (currentChannels == streamChannels) {
                    // If running stream channels match, match we want to stop it
                    m_streamer->stopDataStream(streamType, streamFormat);
                    if (appState.options.vrColorCorrectionEnabled) {
                        // If color correction is enabled, start stream with no data channels to
                        // keep getting camera metadata for exposure adjustment
                        LOG_INFO("Switching to metadata only color stream..");
                        m_streamer->startDataStream(streamType, streamFormat, varjo_ChannelFlag_None);
                    }
                } else {
                    // When color correction is enabled, stop stream and start new stream with data channels
                    LOG_INFO("Switching to color stream with data channels..");
                    assert(appState.options.vrColorCorrectionEnabled);
                    m_streamer->stopDataStream(streamType, streamFormat);
                    m_streamer->startDataStream(streamType, streamFormat, streamChannels);
                }
            } else {
                // No streams running, just ignore
            }
        }

        // Write stream status back to state
        m_appState.options.dataStreamColorEnabled = m_streamer->isStreaming(streamType, streamFormat);
    }

    // Data stream: Cubemap
    if (force || appState.options.dataStreamCubemapEnabled != prevState.options.dataStreamCubemapEnabled) {
        const auto streamType = varjo_StreamType_EnvironmentCubemap;
        const auto streamFormat = varjo_TextureFormat_RGBA16_FLOAT;
        const auto streamChannels = varjo_ChannelFlag_First;

        if (appState.options.dataStreamCubemapEnabled) {
            if (!m_streamer->isStreaming(streamType, streamFormat)) {
                m_streamer->startDataStream(streamType, streamFormat, streamChannels);
            }
        } else {
            if (m_streamer->isStreaming(streamType, streamFormat)) {
                m_streamer->stopDataStream(streamType, streamFormat);
            }
        }

        // Write stream status back to state
        m_appState.options.dataStreamCubemapEnabled = m_streamer->isStreaming(streamType, streamFormat);
    }
}

void AppLogic::update()
{
    // Check for new mixed reality events
    checkEvents();

    // Handle delayed data stream buffers
    m_streamer->handleDelayedBuffers();

    // Sync frame
    m_varjoView->syncFrame();

    // Update frame time
    m_appState.general.frameTime += m_varjoView->getDeltaTime();
    m_appState.general.frameCount = m_varjoView->getFrameNumber();

    // Update scene
    MRScene::UpdateParams updateParams{};

    // Get latest color adjustments
    if (m_appState.options.vrColorCorrectionEnabled && m_appState.options.videoRenderingEnabled) {
        auto exposureAdjustments = m_streamer->getExposureAdjustments();
        if (exposureAdjustments.valid) {
            updateParams.cameraParams.exposureEV = exposureAdjustments.ev;
            updateParams.cameraParams.cameraCalibrationConstant = exposureAdjustments.cameraCalibrationConstant;
            updateParams.cameraParams.wbNormalizationData = exposureAdjustments.wbNormalizationData;
        }
    }

    updateParams.lighting.ambientLight = m_appState.options.ambientLightGainRGB;
    m_scene->update(m_varjoView->getFrameTime(), m_varjoView->getDeltaTime(), m_varjoView->getFrameNumber(), updateParams);

    // Get latest cubemap frame.
    DataStreamer::CubemapFrame cubemapFrame;
    if (m_streamer->getCubemapFrame(cubemapFrame)) {
        m_scene->updateHdrCubemap(cubemapFrame.metadata.width, cubemapFrame.metadata.format, cubemapFrame.metadata.rowStride, cubemapFrame.data.data());
    }

    // Early exit if no frame submit
    if (!m_appState.options.renderVREnabled) {
        // Invalidate frame by submitting empty frame on the first call, then no op
        m_varjoView->invalidateFrame();
        return;
    }

    // Begin frame
    m_varjoView->beginFrame();

    // Render layer
    {
        // Get layer for rendering
        constexpr int layerIndex = 0;
        auto& layer = m_varjoView->getLayer(layerIndex);

        // Setup render params
        MultiLayerView::Layer::SubmitParams submitParams{};
        submitParams.submitColor = m_appState.options.renderVREnabled;
        submitParams.submitDepth = m_appState.options.submitVRDepthEnabled;
        submitParams.depthTestEnabled = m_appState.options.VideoDepthEstimationEnabled;
        submitParams.depthTestRangeEnabled = m_appState.options.vrDepthTestRangeEnabled;
        submitParams.depthTestRangeLimits = {
            0.0, submitParams.depthTestRangeEnabled ? std::max(static_cast<double>(m_appState.options.vrDepthTestRangeValue), 0.0) : 0.0};
        submitParams.chromaKeyEnabled = m_appState.options.chromaKeyingEnabled;
        submitParams.alphaBlend = m_appState.options.videoRenderingEnabled || !m_appState.options.drawVRBackgroundEnabled;

        // Begin layer rendering
        layer.begin(submitParams);

        // Clear frame
        const glm::vec4 opaqueBg(0.15f, 0.5f, 0.6f, 1.0f);
        const glm::vec4 transpBg(0.0f, 0.0f, 0.0f, 0.0f);
        const glm::vec4 clearColor = submitParams.alphaBlend ? transpBg : opaqueBg;
        layer.clear(MultiLayerView::Layer::ClearParams(clearColor));

        // Render frame
        layer.renderScene(*m_scene);

        // End layer rendering
        layer.end();
    }

    // Submit varjo frame with rendered layers
    m_varjoView->endFrame();
}

void AppLogic::onMixedRealityAvailable(bool available, bool forceSetState)
{
    m_appState.general.mrAvailable = available;

    if (available) {
        // Update stuff here if needed

        // Get format for color stream
        m_colorStreamFormat = m_streamer->getFormat(varjo_StreamType_DistortedColor);

        if (m_appState.options.reactToConnectionEvents && !m_appState.options.videoRenderingEnabled) {
            LOG_INFO("Enabling video rendering on MR available event..");
            setVideoRendering(true);
        }

    } else {
        LOG_ERROR("Mixed Reality features not available.");

        // Update stuff here if needed
        if (m_appState.options.reactToConnectionEvents && m_appState.options.videoRenderingEnabled) {
            LOG_INFO("Disabling video rendering on MR unavailable event..");
            setVideoRendering(false);
        }
    }

    // Force set state when MR becomes active
    if (forceSetState) {
        setState(m_appState, true);
    }

    // Update camera status
    m_camera->updateStatus();
}

void AppLogic::checkEvents()
{
    varjo_Bool ret = varjo_False;

    do {
        varjo_Event evt{};
        ret = varjo_PollEvent(m_session, &evt);
        CHECK_VARJO_ERR(m_session);

        if (ret == varjo_True) {
            switch (evt.header.type) {
                case varjo_EventType_MRDeviceStatus: {
                    switch (evt.data.mrDeviceStatus.status) {
                        // Occurs when Mixed Reality features are enabled
                        case varjo_MRDeviceStatus_Connected: {
                            LOG_INFO("EVENT: Mixed reality device status: %s", "Connected");
                            constexpr bool forceSetState = true;
                            onMixedRealityAvailable(true, forceSetState);
                        } break;
                        // Occurs when Mixed Reality features are disabled
                        case varjo_MRDeviceStatus_Disconnected: {
                            LOG_INFO("EVENT: Mixed reality device status: %s", "Disconnected");
                            constexpr bool forceSetState = false;
                            onMixedRealityAvailable(false, forceSetState);
                        } break;
                        default: {
                            // Ignore unknown status.
                        } break;
                    }
                } break;

                case varjo_EventType_MRCameraPropertyChange: {
                    varjo_CameraPropertyMode mode = varjo_MRGetCameraPropertyMode(m_session, evt.data.mrCameraPropertyChange.type);
                    varjo_CameraPropertyValue manualValue = varjo_MRGetCameraPropertyValue(m_session, evt.data.mrCameraPropertyChange.type);
                    std::string cameraPropStr = CameraManager::propertyTypeToString(evt.data.mrCameraPropertyChange.type);
                    std::string modeStr = CameraManager::propertyModeToString(mode);
                    std::string valStr = CameraManager::propertyValueToString(manualValue);
                    LOG_INFO("EVENT: Camera prop changed: %s. mode: %s, value: %s", cameraPropStr.c_str(), modeStr.c_str(), valStr.c_str());
                } break;

                case varjo_EventType_DataStreamStart: {
                    LOG_INFO("EVENT: Data stream started: id=%d", static_cast<int>(evt.data.dataStreamStart.streamId));
                } break;

                case varjo_EventType_DataStreamStop: {
                    LOG_INFO("EVENT: Data stream stopped: id=%d", static_cast<int>(evt.data.dataStreamStop.streamId));
                } break;

                default: {
                    // Ignore unknown event.
                } break;
            }
        }
    } while (ret);
}
