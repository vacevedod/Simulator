// Copyright 2019-2021 Varjo Technologies Oy. All rights reserved.

#pragma once

#include "Globals.hpp"
#include "MultiLayerView.hpp"
//#include "D3D11Renderer.hpp"
#include "GLRenderer.hpp"

namespace VarjoExamples
{
    class GLMultiLayerView;

    //! Layer view implementation for D3D11 renderer
    class GLMultiLayerView final : public MultiLayerView
    {
    public:
        // Layer implementation for D3D11 renderer
        class GLLayer final : public MultiLayerView::Layer
        {
        public:
            //! Layer configuration structure
            struct Config {
                int contextDivider{ 1 };                                          //!< Context texture size divider from full view size
                int focusDivider{ 1 };                                            //!< Context texture size divider from full view size
                varjo_TextureFormat format{ varjo_TextureFormat_R8G8B8A8_SRGB };  //!< Layer texture buffer format
            };

            //! Constructor
            GLLayer(GLMultiLayerView& view, const Config& config);

            //! Destructor
            ~GLLayer();

        private:
            GLMultiLayerView& m_glView;  //!< D3D11 view multi layer view instance
        };

        //! Constructor
        GLMultiLayerView(varjo_Session* session, GLRenderer& renderer);

        //! Constructor with layer config
        GLMultiLayerView(varjo_Session* session, GLRenderer& renderer, const GLLayer::Config& layerConfig);

        //! Constructor with configs for multiple layers
        GLMultiLayerView(varjo_Session* session, GLRenderer& renderer, const std::vector<GLLayer::Config>& layerConfigs);

        //! Destructor
        ~GLMultiLayerView();

        //! Static function for getting DXGI adapter used by Varjo compositor
        //static ComPtr<IDXGIAdapter> getAdapter(varjo_Session* session);

    private:
        GLRenderer& m_glRenderer;  //!< D3D11 renderer instance
    };

}  // namespace VarjoExamples