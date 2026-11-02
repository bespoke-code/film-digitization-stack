//! Dedicated wgpu preview renderer.
//!
//! This adapter owns the GPU pipeline, sampler, uniform buffer and the uploaded
//! texture. It deliberately exposes no egui type: the egui adapter only forwards
//! a rect and a render pass. The same renderer could therefore be driven by a
//! future Slint or Qt adapter.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use stitch_domain::RgbaImage;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct PreviewUniform {
    /// Left, top, right, bottom in normalized device coordinates.
    rect: [f32; 4],
}

struct TextureState {
    bind_group: wgpu::BindGroup,
    _texture: wgpu::Texture,
    width: u32,
    height: u32,
    revision: u64,
}

/// GPU resources for drawing one image into an egui (or other) render pass.
pub struct PreviewRenderer {
    pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    uniform_buffer: wgpu::Buffer,
    texture: Option<TextureState>,
    max_texture_dimension: u32,
    format: wgpu::TextureFormat,
}

impl PreviewRenderer {
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scan-to-img preview shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("preview.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scan-to-img preview bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<PreviewUniform>() as u64,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scan-to-img preview pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("scan-to-img preview pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: sample_count.max(1),
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scan-to-img preview uniform"),
            contents: bytemuck::bytes_of(&PreviewUniform {
                rect: [-1.0, 1.0, 1.0, -1.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("scan-to-img preview sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let limits = device.limits();
        Self {
            pipeline,
            sampler,
            uniform_buffer,
            texture: None,
            max_texture_dimension: limits.max_texture_dimension_2d.max(1),
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
        }
    }

    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    pub fn uploaded_revision(&self) -> Option<u64> {
        self.texture.as_ref().map(|texture| texture.revision)
    }

    pub fn image_size(&self) -> Option<(u32, u32)> {
        self.texture
            .as_ref()
            .map(|texture| (texture.width, texture.height))
    }

    /// Uploads the image if its revision changed and updates the quad rect.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        image: Option<&Arc<RgbaImage>>,
        revision: u64,
        ndc_rect: [f32; 4],
    ) {
        let Some(image) = image else {
            self.texture = None;
            return;
        };

        if self
            .texture
            .as_ref()
            .map(|texture| texture.revision != revision)
            .unwrap_or(true)
        {
            self.upload(device, queue, image, revision);
        }

        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&PreviewUniform { rect: ndc_rect }),
        );
    }

    fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        image: &Arc<RgbaImage>,
        revision: u64,
    ) {
        let upload = if image.width().max(image.height()) > self.max_texture_dimension {
            image.resized_to_fit(self.max_texture_dimension)
        } else {
            image.as_ref().clone()
        };
        let width = upload.width();
        let height = upload.height();

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scan-to-img preview texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            upload.pixels(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self.create_bind_group(device, &view);
        self.texture = Some(TextureState {
            bind_group,
            _texture: texture,
            width,
            height,
            revision,
        });
    }

    fn create_bind_group(
        &self,
        device: &wgpu::Device,
        view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scan-to-img preview bind group"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }

    /// Draws the textured quad into an existing render pass. The caller sets the
    /// scissor rect to the viewport clip.
    pub fn paint(&self, render_pass: &mut wgpu::RenderPass<'static>) {
        let Some(texture) = &self.texture else {
            return;
        };
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &texture.bind_group, &[]);
        render_pass.draw(0..6, 0..1);
    }

    pub fn clear(&mut self) {
        self.texture = None;
    }
}

/// Converts a rect in logical points into normalized device coordinates.
///
/// `rect` is `[min_x, min_y, max_x, max_y]`, `screen` is `[width, height]` in
/// points. The returned rect is `[left, top, right, bottom]` with `y` up.
pub fn rect_to_ndc(rect: [f32; 4], screen: [f32; 2]) -> [f32; 4] {
    let width = screen[0].max(1.0);
    let height = screen[1].max(1.0);
    [
        rect[0] / width * 2.0 - 1.0,
        1.0 - rect[1] / height * 2.0,
        rect[2] / width * 2.0 - 1.0,
        1.0 - rect[3] / height * 2.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_screen_rect_maps_to_clip_space() {
        let ndc = rect_to_ndc([0.0, 0.0, 800.0, 600.0], [800.0, 600.0]);
        assert!((ndc[0] + 1.0).abs() < 1e-6);
        assert!((ndc[1] - 1.0).abs() < 1e-6);
        assert!((ndc[2] - 1.0).abs() < 1e-6);
        assert!((ndc[3] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn centered_half_size_rect() {
        let ndc = rect_to_ndc([200.0, 150.0, 600.0, 450.0], [800.0, 600.0]);
        assert!((ndc[0] + 0.5).abs() < 1e-6);
        assert!((ndc[1] - 0.5).abs() < 1e-6);
        assert!((ndc[2] - 0.5).abs() < 1e-6);
        assert!((ndc[3] + 0.5).abs() < 1e-6);
    }
}
