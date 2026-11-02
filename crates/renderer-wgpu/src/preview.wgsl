struct Uniforms {
    // left, top, right, bottom in normalized device coordinates
    rect: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var image_texture: texture_2d<f32>;
@group(0) @binding(2) var image_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    var corners = array<vec2<f32>, 4>(
        vec2<f32>(uniforms.rect.x, uniforms.rect.y),
        vec2<f32>(uniforms.rect.z, uniforms.rect.y),
        vec2<f32>(uniforms.rect.x, uniforms.rect.w),
        vec2<f32>(uniforms.rect.z, uniforms.rect.w),
    );
    var uvs = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
    );
    var indices = array<u32, 6>(0u, 1u, 2u, 2u, 1u, 3u);

    var output: VertexOutput;
    let corner = corners[indices[index]];
    output.position = vec4<f32>(corner, 0.0, 1.0);
    output.uv = uvs[indices[index]];
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(image_texture, image_sampler, input.uv);
}
