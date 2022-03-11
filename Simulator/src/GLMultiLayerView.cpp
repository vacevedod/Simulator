// Copyright 2019-2021 Varjo Technologies Oy. All rights reserved.

#include "D3D11MultiLayerView.hpp"
#include "GLMultiLayerView.hpp"
//#include <Varjo_d3d11.h>
#include <Varjo_gl.h>

namespace VarjoExamples
{
    GLMultiLayerView::GLLayer::GLLayer(GLMultiLayerView& view, const GLLayer::Config& config)
        : MultiLayerView::Layer(view, config.contextDivider, config.focusDivider)
        , m_glView(view)
    {
        // Create color texture swap chain for DX11
        varjo_SwapChainConfig2 colorConfig{};
        colorConfig.numberOfTextures = 4;
        colorConfig.textureArraySize = 1;
        colorConfig.textureFormat.textureFormat = config.format;

        const auto totalSize = getTotalSize(m_viewports);
        colorConfig.textureWidth = static_cast<int32_t>(totalSize.x);
        colorConfig.textureHeight = static_cast<int32_t>(totalSize.y);

        /*    
        colorConfig.textureWidth = getTotalViewportsWidth();
        colorConfig.textureHeight = getTotalViewportsHeight();
    */

        m_colorSwapChain = varjo_GLCreateSwapChain(m_view.getSession(), m_glView.m_glRenderer.getD3DDevice(), &colorConfig);
        CHECK_VARJO_ERR(m_view.getSession());

        // Create a DX11 render target for each color swap chain texture
        for (int i = 0; i < colorConfig.numberOfTextures; ++i) {
            // Get swap chain textures for render target
            const varjo_Texture colorTexture = varjo_GetSwapChainImage(m_colorSwapChain, i);
            CHECK_VARJO_ERR(m_view.getSession());

            // Create render target instance
            m_colorRenderTargets.emplace_back(std::make_unique<GLRenderer::ColorRenderTarget>(
                m_glView.m_glRenderer.getD3DDevice(), colorConfig.textureWidth, colorConfig.textureHeight, varjo_ToGLTexture(colorTexture)));
        }

        // Create depth texture swap chain for DX11
        varjo_SwapChainConfig2 depthConfig{ colorConfig };
        depthConfig.textureFormat = varjo_DepthTextureFormat_D32_FLOAT;

        m_depthSwapChain = varjo_GLCreateSwapChain(m_view.getSession(), m_glView.m_glRenderer.getD3DDevice(), &depthConfig);
        CHECK_VARJO_ERR(m_view.getSession());

        // Create a DX11 render target for each depth swap chain texture
        for (int i = 0; i < depthConfig.numberOfTextures; ++i) {
            const varjo_Texture depthTexture = varjo_GetSwapChainImage(m_depthSwapChain, i);
            CHECK_VARJO_ERR(m_view.getSession());

            // Create render target instance
            m_depthRenderTargets.emplace_back(std::make_unique<GLRenderer::DepthRenderTarget>(
                m_glView.m_glRenderer.getD3DDevice(), depthConfig.textureWidth, depthConfig.textureHeight, varjo_ToGLTexture(depthTexture)));
        }

        // Setup views
        setupViews();
    }

    GLMultiLayerView::GLLayer::~GLLayer() = default;

    // --------------------------------------------------------------------------

    GLMultiLayerView::GLMultiLayerView(varjo_Session* session, GLRenderer& renderer)
        : MultiLayerView(session, renderer)
        , m_glRenderer(renderer)
    {
        // Create layer instance
        GLLayer::Config layerConfig{};
        m_layers.emplace_back(std::make_unique<GLLayer>(*this, layerConfig));
    }

    GLMultiLayerView::GLMultiLayerView(varjo_Session* session, GLRenderer& renderer, const GLLayer::Config& layerConfig)
        : MultiLayerView(session, renderer)
        , m_glRenderer(renderer)
    {
        // Create layer instance
        m_layers.emplace_back(std::make_unique<GLLayer>(*this, layerConfig));
    }

    GLMultiLayerView::GLMultiLayerView(varjo_Session* session, GLRenderer& renderer, const std::vector<GLLayer::Config>& layerConfigs)
        : MultiLayerView(session, renderer)
        , m_glRenderer(renderer)
    {
        // Create layer instances
        for (const auto& layerConfig : layerConfigs) {
            m_layers.emplace_back(std::make_unique<GLLayer>(*this, layerConfig));
        }
    }

    GLMultiLayerView::~GLMultiLayerView() = default;
    /*
    ComPtr<IDXGIAdapter> GLMultiLayerView::getAdapter(varjo_Session* session)
    {
        varjo_Luid luid = varjo_D3D11GetLuid(session);

        ComPtr<IDXGIFactory> factory = nullptr;
        const HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
        if (SUCCEEDED(hr)) {
            UINT i = 0;
            while (true) {
                ComPtr<IDXGIAdapter> adapter = nullptr;
                if (factory->EnumAdapters(i++, &adapter) == DXGI_ERROR_NOT_FOUND) break;
                DXGI_ADAPTER_DESC desc;
                if (SUCCEEDED(adapter->GetDesc(&desc)) && desc.AdapterLuid.HighPart == luid.high && desc.AdapterLuid.LowPart == luid.low) {
                    return adapter;
                }
            }
        }

        LOG_WARNING("Could not get DXGI adapter.");
        return nullptr;
    }*/

}  // namespace VarjoExamples
