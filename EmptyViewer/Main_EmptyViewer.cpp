#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

#define GLFW_INCLUDE_GLU
#define GLFW_DLL
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>        


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace glm;

int Width = 512;  
int Height = 512; 
std::vector<vec3> OutputImage; // ȭ�鿡 �׷��� �ȼ� ���� ����
std::vector<float> ZBuffer;    // ���� ����

struct Vertex {
    float x, y, z;//3d��ǥ
	float nx, ny, nz;//���� ����
};

namespace SphereGenerator {
    std::vector<Vertex> gVertices;    // ���� ���� ������ ����
    std::vector<int> gIndexBuffer; // �ﰢ���� �����ϴ� ���� �ε������� ������ ����
    int gNumVertices_generated = 0;   // ������ ���� ���� ��
    int gNumTriangles_generated = 0;  // ������ ���� �ﰢ�� �� 

    void create_scene_geometry() {
        gVertices.clear();
        gIndexBuffer.clear();

        int width_segments = 32;  // ���� �����ϴ� �浵�� ��
        int height_segments = 16; // ���� �����ϴ� ������ ��

        int expected_num_vertices = (height_segments - 2) * width_segments + 2;
        int expected_num_triangles = (height_segments - 2) * (width_segments - 1) * 2;

        gVertices.reserve(expected_num_vertices); // �޸� �̸� �Ҵ�
        gIndexBuffer.reserve(expected_num_triangles * 3);

        int t_vtx = 0; //���� �迭 ī����

        // ���� �κ� ���� ���� (�ϱذ� ���� ����)
        for (int j = 1; j < height_segments - 1; ++j) { // height_segments-2 ���� ������ ����
            for (int i = 0; i < width_segments; ++i) {   // �� ���������� width_segments ���� ���� ����
                //�� -> ���� ��ǥ ��ȯ
                float theta = static_cast<float>(j) / (height_segments - 1) * static_cast<float>(M_PI);
                float phi = static_cast<float>(i) / (width_segments - 1) * 2.0f * static_cast<float>(M_PI); // width_segments-1�� ������ �������� ó���� �����

                Vertex v;
                v.x = sinf(theta) * cosf(phi); //�� -> ���� ��ǥ ��ȯ
                v.y = cosf(theta);
                v.z = -sinf(theta) * sinf(phi); // Z�� ������ -�� ���� 

				glm::vec3 normal = glm::normalize(glm::vec3(v.x, v.y, v.z)); // ���� ���� ���
				v.nx = normal.x;// ���� ���͸� ������ ����
				v.ny = normal.y;
				v.nz = normal.z;
                gVertices.push_back(v);
                t_vtx++;
            }
        }

        // �ϱ� ���� (0, 1, 0)
        gVertices.push_back({ 0.0f, 1.0f, 0.0f,0.0f,1.0f,0.0f });
        int northPoleIndex = t_vtx; // �ϱ����� �ε���
        t_vtx++;

        // ���� ���� (0, -1, 0)
        gVertices.push_back({ 0.0f, -1.0f, 0.0f,0.0f,-1.0f,0.0f});
        int southPoleIndex = t_vtx; // �������� �ε���
        t_vtx++;

        gNumVertices_generated = t_vtx;


        int t_idx = 0; // �ε��� ���ۿ� ī����

        for (int j = 0; j < height_segments - 3; ++j) { // 0 ���� height_segments-4����
            for (int i = 0; i < width_segments; ++i) { 
                int v0 = j * width_segments + i;
                int v1 = j * width_segments + (i + 1) % width_segments; // ������ ������ ù ���� ����
                int v2 = (j + 1) * width_segments + i;
                int v3 = (j + 1) * width_segments + (i + 1) % width_segments; 

                // ù ��° �ﰢ�� (v0, v3, v1) - �ݽð� ���� (������ �� �ٱ������� ����)
                gIndexBuffer.push_back(v0);
                gIndexBuffer.push_back(v3);
                gIndexBuffer.push_back(v1);
                t_idx += 3;

                // �� ��° �ﰢ�� (v0, v2, v3) - �ݽð� ����
                gIndexBuffer.push_back(v0);
                gIndexBuffer.push_back(v2);
                gIndexBuffer.push_back(v3);
                t_idx += 3;
            }
        }

        // �ϱ� ĸ (�ﰢ�� ��)
        // �ϱ����� ù ��° ������(�ε��� 0 ~ width_segments-1)�� ����
        for (int i = 0; i < width_segments - 1; ++i) {
            gIndexBuffer.push_back(northPoleIndex);
            gIndexBuffer.push_back(i);
            gIndexBuffer.push_back((i + 1)%width_segments);
            t_idx += 3;
        }

        // ���� ĸ (�ﰢ�� ��)
        // �������� ������ ������ ������(�ε��� (height_segments-3)*width_segments ~ (height_segments-2)*width_segments-1)�� ����
        int lastLatStartIdx = (height_segments - 3) * width_segments;
        for (int i = 0; i < width_segments - 1; ++i) {
            gIndexBuffer.push_back(southPoleIndex);
            gIndexBuffer.push_back(lastLatStartIdx + (i + 1)%width_segments);
            gIndexBuffer.push_back(lastLatStartIdx + i);
            t_idx += 3;
        }
        gNumTriangles_generated = t_idx / 3; // �ﰢ�� ���� �ε��� �� / 3
    }
}

// p: �ȼ� �߽� ��ǥ, a,b,c: �ﰢ�� ������ 2D ȭ�� ��ǥ
// �ﰢ�� ���� p�� ���� ��ǥ���(alpha, beta, gamma) 
vec3 barycentric_coords(vec2 p, vec2 a, vec2 b, vec2 c) {
    vec2 v0 = b - a, v1 = c - a, v2 = p - a;
    float d00 = dot(v0, v0);
    float d01 = dot(v0, v1);
    float d11 = dot(v1, v1);
    float d20 = dot(v2, v0);
    float d21 = dot(v2, v1);
    float denom = d00 * d11 - d01 * d01; //�и�

    if (std::abs(denom) < 1e-5) { // �и� 0�� ������ �ﰢ���� �� �Ǵ� ��
        return vec3(-1.0f, -1.0f, -1.0f); // ��ȿ���� ���� ��ǥ ��ȯ
    }

    float v_coord = (d11 * d20 - d01 * d21) / denom; // beta (b�� ���� ����ġ)
    float w_coord = (d00 * d21 - d01 * d20) / denom; // gamma (c�� ���� ����ġ)
    float u_coord = 1.0f - v_coord - w_coord;      // alpha (a�� ���� ����ġ)
    return vec3(u_coord, v_coord, w_coord);
    //�ȼ��� �ﰢ�� ���ο� ������ u,v,w�� 0���� ũ�� ���� 1��
}
vec3 calculate_blinn_phong_shading(
    const vec3& frag_pos_world,   //  ���� ��ǥ
    const vec3& normal_world,     // ���� ���� ����
    const vec3& eye_pos_world,    // ī�޶��� ���� ��ǥ
    const vec3& light_pos_world,  // ������ ���� ��ǥ
    const vec3& light_intensity, // ������ ����
    float ambient_intensity_val, // �ֺ��� ����
    const vec3& k_a, const vec3& k_d, const vec3& k_s, float p_shininess // �ݻ� ���
) {
    vec3 ambient_color = k_a * ambient_intensity_val;

    vec3 N = normalize(normal_world);
    vec3 L = normalize(light_pos_world - frag_pos_world); // ���� ���� ����
    vec3 V = normalize(eye_pos_world - frag_pos_world); // �ü� ���� ����
    vec3 H = normalize(L + V);                          // ���� ����

    // Diffuse
    float diff_dot = std::max(dot(N, L), 0.0f);
    vec3 diffuse_color = k_d * light_intensity * diff_dot;

    // Specular
    float spec_dot = std::max(dot(N, H), 0.0f);
    vec3 specular_color = k_s * light_intensity * pow(spec_dot, p_shininess);

    return ambient_color + diffuse_color + specular_color;
}

void render_rasterized() {
	OutputImage.assign(Width * Height, vec3(0.0f, 0.0f, 0.0f)); // ���� �������� �ʱ�ȭ
    ZBuffer.assign(Width * Height, std::numeric_limits<float>::infinity()); // ���� ���� �ʱ�ȭ (���� �� ��)
    // ���� �Ӽ�
    const vec3 ka(0.0f, 1.0f, 0.0f);
    const vec3 kd(0.0f, 0.5f, 0.0f);
    const vec3 ks(0.5f, 0.5f, 0.5f);
    const float shininess_p = 32.0f;

    // ���� �Ӽ�
    const vec3 light_pos_world(-4.0f, 4.0f, -3.0f); // ���� ��ǥ�� ����
    const vec3 light_intensity(1.0f, 1.0f, 1.0f);  // ���� �����
	const float ambient_intensity = 0.2f;// �ֺ���

    const vec3 eye_pos_world(0.0f, 0.0f, 0.0f);//ī�޶�

    // 1. ��ȯ ��� ����
    // �𵨸� ��ȯ: ���� �� -> �߽� (0,0,-7), ������ 2
    mat4 model_matrix = mat4(1.0f);
    model_matrix = translate(model_matrix, vec3(0.0f, 0.0f, -7.0f));
    model_matrix = scale(model_matrix, vec3(2.0f, 2.0f, 2.0f));

    // 2. ī�޶�(��) ��ȯ: eye (0,0,0), u=(1,0,0), v=(0,1,0), w=(0,0,1) (ī�޶�� -w ������ ��)
    mat4 view_matrix = lookAt(eye_pos_world, vec3(0.0f, 0.0f, -1.0f), vec3(0.0f, 1.0f, 0.0f));

    // 3. ���� ��ȯ: l=-0.1, r=0.1, b=-0.1, t=0.1, n=-0.1, f=-1000 
    // nearVal�� farVal�� ī�޶�κ����� �Ÿ� (���)
    // ������ n, f�� ī�޶� ���������� z��ǥ�� �ؼ�. ī�޶� -z�� ���Ƿ�,
    // near plane z_cam = -0.1 => near_val = 0.1
    // far  plane z_cam = -1000 => far_val  = 1000
    float frustum_l = -0.1f, frustum_r = 0.1f, frustum_b = -0.1f, frustum_t = 0.1f;
    float frustum_n = 0.1f;  
    float frustum_f = 1000.0f; 
    mat4 projection_matrix = frustum(frustum_l, frustum_r, frustum_b, frustum_t, frustum_n, frustum_f);
    
    // 4.����Ʈ ��ȯ: NDC (-1~1) -> ȭ�� ��ǥ (0~Width, 0~Height), Z (0~1)
    mat4 viewport_matrix = mat4(1.0f);
    // ���� NDC�� [0,1] ������ ������ �� �̵�
    viewport_matrix = translate(viewport_matrix, vec3(0.5f, 0.5f, 0.5f));
    viewport_matrix = scale(viewport_matrix, vec3(0.5f, 0.5f, 0.5f));
    // �� ���� ȭ�� ũ�⿡ �°� ������ �� �̵�
    viewport_matrix = scale(viewport_matrix, vec3(static_cast<float>(Width), static_cast<float>(Height), 1.0f));
   
    // M_vp = [ W/2  0    0    (W-1)/2 ]
    //        [ 0    H/2  0    (H-1)/2 ]
    //        [ 0    0    1/2  1/2     ]  (NDC z�� 0~1�� ����)
    //        [ 0    0    0    1       ]
    // (W-1)/2, (H-1)/2�� �ȼ� �߽��� 0.0���� ���� ��. 0.5 �������̸� (W/2, H/2)
    viewport_matrix = mat4(
        Width / 2.0f, 0.0f, 0.0f, 0.0f,
        0.0f, Height / 2.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.5f, 0.0f, // Z: [-1, 1] -> [0, 1]
        Width / 2.0f, Height / 2.0f, 0.5f, 1.0f
    );
	mat3 normal_transform_matrix = transpose(inverse(mat3(model_matrix))); // ���� ��ȯ ��� 

    // �ﰢ�� ������ȭ
    for (size_t i = 0; i < SphereGenerator::gIndexBuffer.size(); i += 3) {
        // �ﰢ���� �� ���� �ε���
        int idx0 = SphereGenerator::gIndexBuffer[i];
        int idx1 = SphereGenerator::gIndexBuffer[i + 1];
        int idx2 = SphereGenerator::gIndexBuffer[i + 2];

        // ���� ���� ��ǥ (�� ����)
        Vertex v_orig[] = {
            SphereGenerator::gVertices[idx0],
            SphereGenerator::gVertices[idx1],
            SphereGenerator::gVertices[idx2]
        };
        vec3 vert_pos_world[3]; // ���� ���� ���� ��ǥ
        vec3 vert_normal_world[3]; // ���� ���� ���� ����

        for (int k = 0; k < 3; ++k) {
            // ���� ���� ���� ��ǥ
            vec4 p_model = vec4(v_orig[k].x, v_orig[k].y, v_orig[k].z, 1.0f); //�� ���� ��ǥ
            vec4 p_world = model_matrix * p_model; //���� ���� ��ǥ
            vert_pos_world[k] = vec3(p_world); // ���� ���� ���� ��ǥ

            vec3 n_model = vec3(v_orig[k].nx, v_orig[k].ny, v_orig[k].nz); // �� ���� ���� ����
            vert_normal_world[k] = normalize(normal_transform_matrix * n_model); // ���� ���� ��ȯ

            vec4 p_clip[3];    // Ŭ�� ���� ��ǥ
            vec3 p_ndc[3];     // ����ȭ�� ��ġ ��ǥ (NDC)
            vec2 p_screen[3];  // ȭ�� ���� ��ǥ (����Ʈ ��ȯ �� x,y)
            float z_screen[3]; // ȭ�� ���� ���� (����Ʈ ��ȯ �� z)
            float w_clip[3];   // Ŭ�� ���� w �� 

            for (int k = 0; k < 3; ++k) {
                //vec4 p_model = vec4(v_orig[k].x, v_orig[k].y, v_orig[k].z, 1.0f); //�� ���� ��ǥ
                //vec4 p_world = model_matrix * p_model; //���� ���� ��ǥ
                vec4 p_camera = view_matrix * vec4(vert_pos_world[k],1.0f); //ī�޶� ���� ��ǥ
                p_clip[k] = projection_matrix * p_camera;//Ŭ�� ���� ��ǥ
                w_clip[k] = p_clip[k].w;// Ŭ�� ���� w ��

            }

            for (int k = 0; k < 3; ++k) {
                p_ndc[k] = vec3(p_clip[k]) / w_clip[k]; //���̰������� ���� ǥ��

                // ����Ʈ ��ȯ
                vec4 p_screen_h = viewport_matrix * vec4(p_ndc[k], 1.0f);//ȭ�� ��ǥ�� ��ȯ
                p_screen[k] = vec2(p_screen_h);// [0, Width] x [0, Height] ������ ȭ�� ��ǥ
                z_screen[k] = p_screen_h.z; // [0, 1] ������ ���� ��
            }

            // �ٿ�� �ڽ� ���
            int min_x = std::max(0, static_cast<int>(std::floor(std::min({ p_screen[0].x, p_screen[1].x, p_screen[2].x }))));//x �ּ�
            int max_x = std::min(Width - 1, static_cast<int>(std::ceil(std::max({ p_screen[0].x, p_screen[1].x, p_screen[2].x }))));//x�ִ�
            int min_y = std::max(0, static_cast<int>(std::floor(std::min({ p_screen[0].y, p_screen[1].y, p_screen[2].y }))));//y �ּ�
            int max_y = std::min(Height - 1, static_cast<int>(std::ceil(std::max({ p_screen[0].y, p_screen[1].y, p_screen[2].y }))));//y �ִ�

            for (int y_px = min_y; y_px <= max_y; ++y_px) {
                for (int x_px = min_x; x_px <= max_x; ++x_px) {
                    vec2 pixel_center(static_cast<float>(x_px) + 0.5f, static_cast<float>(y_px) + 0.5f);// �ȼ� �߽� ��ǥ
                    vec3 bc = barycentric_coords(pixel_center, p_screen[0], p_screen[1], p_screen[2]);// �ٸ���Ʈ�� ��ǥ ���

                    // �ٸ���Ʈ�� ��ǥ�� ��� 0 �̻��̸� �ȼ��� �ﰢ�� ���ο� ����
                    if (bc.x >= 0.0f && bc.y >= 0.0f && bc.z >= 0.0f) {
                        // ���� ���� ���� ����
                        float w_inv_interpolated = bc.x / w_clip[0] + bc.y / w_clip[1] + bc.z / w_clip[2];//������ 1/w

                        float depth_ndc_interpolated = (bc.x * p_ndc[0].z / w_clip[0] +
                            bc.y * p_ndc[1].z / w_clip[1] +
                            bc.z * p_ndc[2].z / w_clip[2]) / w_inv_interpolated; // ������ ���� ��

                        // ������ NDC ���� ��(-1~1)�� [0, 1] ������ ��ȯ 
                        float current_depth = depth_ndc_interpolated * 0.5f + 0.5f;

                        if (current_depth < ZBuffer[y_px * Width + x_px] && current_depth >= 0.0f && current_depth <= 1.0f) { // ���� �˻� �� ���� Ȯ�� //�����ȼ��� �����ȼ����� ������ ����
                            ZBuffer[y_px * Width + x_px] = current_depth;
                            vec3 pos_numerator = bc.x * vert_pos_world[0] / w_clip[0] +
                                bc.y * vert_pos_world[1] / w_clip[1] +
                                bc.z * vert_pos_world[2] / w_clip[2]; // ������ ����
                            vec3 frag_pos_world_interpolated;
							frag_pos_world_interpolated = pos_numerator / w_inv_interpolated; // ������ ����
                            
							vec3 numerator = bc.x * vert_normal_world[0] / w_clip[0] +
								bc.y * vert_normal_world[1] / w_clip[1] +
								bc.z * vert_normal_world[2] / w_clip[2]; // ������ ���� ����
							vec3 world_interpolated;
							world_interpolated = numerator / w_inv_interpolated; // ������ ���� ����
							vec3 normal_world_interpolated = normalize(world_interpolated); // ���� ���� ����ȭ

							vec3 Phong_color = calculate_blinn_phong_shading(
								frag_pos_world_interpolated,
								normal_world_interpolated,   
								eye_pos_world,              
								light_pos_world,             
								light_intensity,             
								ambient_intensity,           
								ka, kd, ks, shininess_p       
							);

                            vec3 color_before_gamma = Phong_color; // ������ ���� ���
                            vec3 final_color;
                            final_color.r = pow(std::max(0.0f, color_before_gamma.r), 1.0f / 2.2f);
                            final_color.g = pow(std::max(0.0f, color_before_gamma.g), 1.0f / 2.2f);
                            final_color.b = pow(std::max(0.0f, color_before_gamma.b), 1.0f / 2.2f);
                            final_color = glm::clamp(final_color, 0.0f, 1.0f);
                            OutputImage[y_px * Width + x_px] = final_color;
                        }
                    }
                }
            }
        }
    }
        }



//ȭ�� ǥ��
void resize_callback(GLFWwindow* window, int nw, int nh) {
    if (nw <= 0 || nh <= 0) return; 
    Width = nw;
    Height = nh;
    glViewport(0, 0, nw, nh);       
    glMatrixMode(GL_PROJECTION);    
    glLoadIdentity();
    glOrtho(0.0, static_cast<double>(Width),
        0.0, static_cast<double>(Height),
        1.0, -1.0);
    OutputImage.resize(static_cast<size_t>(Width) * Height); 
    ZBuffer.resize(static_cast<size_t>(Width) * Height);     // Z���� ũ�� ����
}

int main(int argc, char* argv[]) {
    GLFWwindow* window;

    if (!glfwInit())
        return -1;

    window = glfwCreateWindow(Width, Height, "CG_hw6", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1); 
    glPixelStorei(GL_PACK_ALIGNMENT, 1);

    glfwSetFramebufferSizeCallback(window, resize_callback); 
    resize_callback(window, Width, Height); 

    // �� ������ ����
    SphereGenerator::create_scene_geometry();
    if (SphereGenerator::gVertices.empty() || SphereGenerator::gIndexBuffer.empty()) {
        std::cerr << "Error: Failed to create sphere geometry." << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }
    std::cout << "Sphere Geometry: Vertices=" << SphereGenerator::gNumVertices_generated
        << ", Triangles=" << SphereGenerator::gNumTriangles_generated << std::endl;

    while (!glfwWindowShouldClose(window)) {
        // ������ȭ ������ ���� (�� ������ �ٽ� ���)
        render_rasterized();

        glClear(GL_COLOR_BUFFER_BIT);
        glDrawPixels(Width, Height, GL_RGB, GL_FLOAT, OutputImage.data());

        glfwSwapBuffers(window); 
        glfwPollEvents();        

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS ||
            glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GL_TRUE);
        }
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}