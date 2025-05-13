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
std::vector<vec3> OutputImage; // 화면에 그려질 픽셀 색상 버퍼
std::vector<float> ZBuffer;    // 깊이 버퍼

struct Vertex {
    float x, y, z;//3d좌표
	float nx, ny, nz;//법선 벡터
};

namespace SphereGenerator {
    std::vector<Vertex> gVertices;    // 구의 정점 저장할 벡터
    std::vector<int> gIndexBuffer; // 삼각형을 구성하는 정점 인덱스들을 저장할 벡터
    int gNumVertices_generated = 0;   // 생성된 실제 정점 수
    int gNumTriangles_generated = 0;  // 생성된 실제 삼각형 수 

    void create_scene_geometry() {
        gVertices.clear();
        gIndexBuffer.clear();

        int width_segments = 32;  // 구를 분할하는 경도선 수
        int height_segments = 16; // 구를 분할하는 위도선 수

        int expected_num_vertices = (height_segments - 2) * width_segments + 2;
        int expected_num_triangles = (height_segments - 2) * (width_segments - 1) * 2;

        gVertices.reserve(expected_num_vertices); // 메모리 미리 할당
        gIndexBuffer.reserve(expected_num_triangles * 3);

        int t_vtx = 0; //정점 배열 카운터

        // 몸통 부분 정점 생성 (북극과 남극 제외)
        for (int j = 1; j < height_segments - 1; ++j) { // height_segments-2 개의 위도선 생성
            for (int i = 0; i < width_segments; ++i) {   // 각 위도선마다 width_segments 개의 정점 생성
                //구 -> 직교 좌표 변환
                float theta = static_cast<float>(j) / (height_segments - 1) * static_cast<float>(M_PI);
                float phi = static_cast<float>(i) / (width_segments - 1) * 2.0f * static_cast<float>(M_PI); // width_segments-1로 나눠야 마지막과 처음이 연결됨

                Vertex v;
                v.x = sinf(theta) * cosf(phi); //구 -> 직교 좌표 변환
                v.y = cosf(theta);
                v.z = -sinf(theta) * sinf(phi); // Z축 방향을 -로 설정 

				glm::vec3 normal = glm::normalize(glm::vec3(v.x, v.y, v.z)); // 법선 벡터 계산
				v.nx = normal.x;// 법선 벡터를 정점에 저장
				v.ny = normal.y;
				v.nz = normal.z;
                gVertices.push_back(v);
                t_vtx++;
            }
        }

        // 북극 정점 (0, 1, 0)
        gVertices.push_back({ 0.0f, 1.0f, 0.0f,0.0f,1.0f,0.0f });
        int northPoleIndex = t_vtx; // 북극점의 인덱스
        t_vtx++;

        // 남극 정점 (0, -1, 0)
        gVertices.push_back({ 0.0f, -1.0f, 0.0f,0.0f,-1.0f,0.0f});
        int southPoleIndex = t_vtx; // 남극점의 인덱스
        t_vtx++;

        gNumVertices_generated = t_vtx;


        int t_idx = 0; // 인덱스 버퍼용 카운터

        for (int j = 0; j < height_segments - 3; ++j) { // 0 부터 height_segments-4까지
            for (int i = 0; i < width_segments; ++i) { 
                int v0 = j * width_segments + i;
                int v1 = j * width_segments + (i + 1) % width_segments; // 마지막 정점과 첫 정점 연결
                int v2 = (j + 1) * width_segments + i;
                int v3 = (j + 1) * width_segments + (i + 1) % width_segments; 

                // 첫 번째 삼각형 (v0, v3, v1) - 반시계 방향 (법선이 구 바깥쪽으로 가게)
                gIndexBuffer.push_back(v0);
                gIndexBuffer.push_back(v3);
                gIndexBuffer.push_back(v1);
                t_idx += 3;

                // 두 번째 삼각형 (v0, v2, v3) - 반시계 방향
                gIndexBuffer.push_back(v0);
                gIndexBuffer.push_back(v2);
                gIndexBuffer.push_back(v3);
                t_idx += 3;
            }
        }

        // 북극 캡 (삼각형 팬)
        // 북극점과 첫 번째 위도선(인덱스 0 ~ width_segments-1)을 연결
        for (int i = 0; i < width_segments - 1; ++i) {
            gIndexBuffer.push_back(northPoleIndex);
            gIndexBuffer.push_back(i);
            gIndexBuffer.push_back((i + 1)%width_segments);
            t_idx += 3;
        }

        // 남극 캡 (삼각형 팬)
        // 남극점과 몸통의 마지막 위도선(인덱스 (height_segments-3)*width_segments ~ (height_segments-2)*width_segments-1)을 연결
        int lastLatStartIdx = (height_segments - 3) * width_segments;
        for (int i = 0; i < width_segments - 1; ++i) {
            gIndexBuffer.push_back(southPoleIndex);
            gIndexBuffer.push_back(lastLatStartIdx + (i + 1)%width_segments);
            gIndexBuffer.push_back(lastLatStartIdx + i);
            t_idx += 3;
        }
        gNumTriangles_generated = t_idx / 3; // 삼각형 수는 인덱스 수 / 3
    }
}

// p: 픽셀 중심 좌표, a,b,c: 삼각형 정점의 2D 화면 좌표
// 삼각형 내부 p에 대한 좌표계산(alpha, beta, gamma) 
vec3 barycentric_coords(vec2 p, vec2 a, vec2 b, vec2 c) {
    vec2 v0 = b - a, v1 = c - a, v2 = p - a;
    float d00 = dot(v0, v0);
    float d01 = dot(v0, v1);
    float d11 = dot(v1, v1);
    float d20 = dot(v2, v0);
    float d21 = dot(v2, v1);
    float denom = d00 * d11 - d01 * d01; //분모

    if (std::abs(denom) < 1e-5) { // 분모가 0에 가까우면 삼각형이 선 또는 점
        return vec3(-1.0f, -1.0f, -1.0f); // 유효하지 않은 좌표 반환
    }

    float v_coord = (d11 * d20 - d01 * d21) / denom; // beta (b에 대한 가중치)
    float w_coord = (d00 * d21 - d01 * d20) / denom; // gamma (c에 대한 가중치)
    float u_coord = 1.0f - v_coord - w_coord;      // alpha (a에 대한 가중치)
    return vec3(u_coord, v_coord, w_coord);
    //픽셀이 삼각형 내부에 있으면 u,v,w가 0보다 크고 합은 1임
}
vec3 calculate_blinn_phong_shading(
    const vec3& frag_pos_world,   //  월드 좌표
    const vec3& normal_world,     // 월드 공간 법선
    const vec3& eye_pos_world,    // 카메라의 월드 좌표
    const vec3& light_pos_world,  // 광원의 월드 좌표
    const vec3& light_intensity, // 광원의 세기
    float ambient_intensity_val, // 주변광 세기
    const vec3& k_a, const vec3& k_d, const vec3& k_s, float p_shininess // 반사 계수
) {
    vec3 ambient_color = k_a * ambient_intensity_val;

    vec3 N = normalize(normal_world);
    vec3 L = normalize(light_pos_world - frag_pos_world); // 광원 방향 벡터
    vec3 V = normalize(eye_pos_world - frag_pos_world); // 시선 방향 벡터
    vec3 H = normalize(L + V);                          // 하프 벡터

    // Diffuse
    float diff_dot = std::max(dot(N, L), 0.0f);
    vec3 diffuse_color = k_d * light_intensity * diff_dot;

    // Specular
    float spec_dot = std::max(dot(N, H), 0.0f);
    vec3 specular_color = k_s * light_intensity * pow(spec_dot, p_shininess);

    return ambient_color + diffuse_color + specular_color;
}

void render_rasterized() {
	OutputImage.assign(Width * Height, vec3(0.0f, 0.0f, 0.0f)); // 배경색 검정으로 초기화
    ZBuffer.assign(Width * Height, std::numeric_limits<float>::infinity()); // 깊이 버퍼 초기화 (가장 먼 값)
    // 재질 속성
    const vec3 ka(0.0f, 1.0f, 0.0f);
    const vec3 kd(0.0f, 0.5f, 0.0f);
    const vec3 ks(0.5f, 0.5f, 0.5f);
    const float shininess_p = 32.0f;

    // 광원 속성
    const vec3 light_pos_world(-4.0f, 4.0f, -3.0f); // 월드 좌표계 기준
    const vec3 light_intensity(1.0f, 1.0f, 1.0f);  // 단위 백색광
	const float ambient_intensity = 0.2f;// 주변광

    const vec3 eye_pos_world(0.0f, 0.0f, 0.0f);//카메라

    // 1. 변환 행렬 설정
    // 모델링 변환: 단위 구 -> 중심 (0,0,-7), 반지름 2
    mat4 model_matrix = mat4(1.0f);
    model_matrix = translate(model_matrix, vec3(0.0f, 0.0f, -7.0f));
    model_matrix = scale(model_matrix, vec3(2.0f, 2.0f, 2.0f));

    // 2. 카메라(뷰) 변환: eye (0,0,0), u=(1,0,0), v=(0,1,0), w=(0,0,1) (카메라는 -w 방향을 봄)
    mat4 view_matrix = lookAt(eye_pos_world, vec3(0.0f, 0.0f, -1.0f), vec3(0.0f, 1.0f, 0.0f));

    // 3. 투영 변환: l=-0.1, r=0.1, b=-0.1, t=0.1, n=-0.1, f=-1000 
    // nearVal과 farVal은 카메라로부터의 거리 (양수)
    // 과제의 n, f는 카메라 공간에서의 z좌표로 해석. 카메라가 -z를 보므로,
    // near plane z_cam = -0.1 => near_val = 0.1
    // far  plane z_cam = -1000 => far_val  = 1000
    float frustum_l = -0.1f, frustum_r = 0.1f, frustum_b = -0.1f, frustum_t = 0.1f;
    float frustum_n = 0.1f;  
    float frustum_f = 1000.0f; 
    mat4 projection_matrix = frustum(frustum_l, frustum_r, frustum_b, frustum_t, frustum_n, frustum_f);
    
    // 4.뷰포트 변환: NDC (-1~1) -> 화면 좌표 (0~Width, 0~Height), Z (0~1)
    mat4 viewport_matrix = mat4(1.0f);
    // 먼저 NDC를 [0,1] 범위로 스케일 및 이동
    viewport_matrix = translate(viewport_matrix, vec3(0.5f, 0.5f, 0.5f));
    viewport_matrix = scale(viewport_matrix, vec3(0.5f, 0.5f, 0.5f));
    // 그 다음 화면 크기에 맞게 스케일 및 이동
    viewport_matrix = scale(viewport_matrix, vec3(static_cast<float>(Width), static_cast<float>(Height), 1.0f));
   
    // M_vp = [ W/2  0    0    (W-1)/2 ]
    //        [ 0    H/2  0    (H-1)/2 ]
    //        [ 0    0    1/2  1/2     ]  (NDC z를 0~1로 매핑)
    //        [ 0    0    0    1       ]
    // (W-1)/2, (H-1)/2는 픽셀 중심을 0.0으로 맞출 때. 0.5 오프셋이면 (W/2, H/2)
    viewport_matrix = mat4(
        Width / 2.0f, 0.0f, 0.0f, 0.0f,
        0.0f, Height / 2.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.5f, 0.0f, // Z: [-1, 1] -> [0, 1]
        Width / 2.0f, Height / 2.0f, 0.5f, 1.0f
    );
	mat3 normal_transform_matrix = transpose(inverse(mat3(model_matrix))); // 법선 변환 행렬 

    // 삼각형 래스터화
    for (size_t i = 0; i < SphereGenerator::gIndexBuffer.size(); i += 3) {
        // 삼각형의 세 정점 인덱스
        int idx0 = SphereGenerator::gIndexBuffer[i];
        int idx1 = SphereGenerator::gIndexBuffer[i + 1];
        int idx2 = SphereGenerator::gIndexBuffer[i + 2];

        // 원본 정점 좌표 (모델 공간)
        Vertex v_orig[] = {
            SphereGenerator::gVertices[idx0],
            SphereGenerator::gVertices[idx1],
            SphereGenerator::gVertices[idx2]
        };
        vec3 vert_pos_world[3]; // 월드 공간 정점 좌표
        vec3 vert_normal_world[3]; // 월드 공간 법선 벡터

        for (int k = 0; k < 3; ++k) {
            // 월드 공간 정점 좌표
            vec4 p_model = vec4(v_orig[k].x, v_orig[k].y, v_orig[k].z, 1.0f); //모델 공간 좌표
            vec4 p_world = model_matrix * p_model; //월드 공간 좌표
            vert_pos_world[k] = vec3(p_world); // 월드 공간 정점 좌표

            vec3 n_model = vec3(v_orig[k].nx, v_orig[k].ny, v_orig[k].nz); // 모델 공간 법선 벡터
            vert_normal_world[k] = normalize(normal_transform_matrix * n_model); // 법선 벡터 변환

            vec4 p_clip[3];    // 클립 공간 좌표
            vec3 p_ndc[3];     // 정규화된 장치 좌표 (NDC)
            vec2 p_screen[3];  // 화면 공간 좌표 (뷰포트 변환 후 x,y)
            float z_screen[3]; // 화면 공간 깊이 (뷰포트 변환 후 z)
            float w_clip[3];   // 클립 공간 w 값 

            for (int k = 0; k < 3; ++k) {
                //vec4 p_model = vec4(v_orig[k].x, v_orig[k].y, v_orig[k].z, 1.0f); //모델 공간 좌표
                //vec4 p_world = model_matrix * p_model; //월드 공간 좌표
                vec4 p_camera = view_matrix * vec4(vert_pos_world[k],1.0f); //카메라 공간 좌표
                p_clip[k] = projection_matrix * p_camera;//클립 공간 좌표
                w_clip[k] = p_clip[k].w;// 클립 공간 w 값

            }

            for (int k = 0; k < 3; ++k) {
                p_ndc[k] = vec3(p_clip[k]) / w_clip[k]; //깊이값나눠서 원근 표현

                // 뷰포트 변환
                vec4 p_screen_h = viewport_matrix * vec4(p_ndc[k], 1.0f);//화면 좌표로 변환
                p_screen[k] = vec2(p_screen_h);// [0, Width] x [0, Height] 범위의 화면 좌표
                z_screen[k] = p_screen_h.z; // [0, 1] 범위의 깊이 값
            }

            // 바운딩 박스 계산
            int min_x = std::max(0, static_cast<int>(std::floor(std::min({ p_screen[0].x, p_screen[1].x, p_screen[2].x }))));//x 최소
            int max_x = std::min(Width - 1, static_cast<int>(std::ceil(std::max({ p_screen[0].x, p_screen[1].x, p_screen[2].x }))));//x최대
            int min_y = std::max(0, static_cast<int>(std::floor(std::min({ p_screen[0].y, p_screen[1].y, p_screen[2].y }))));//y 최소
            int max_y = std::min(Height - 1, static_cast<int>(std::ceil(std::max({ p_screen[0].y, p_screen[1].y, p_screen[2].y }))));//y 최대

            for (int y_px = min_y; y_px <= max_y; ++y_px) {
                for (int x_px = min_x; x_px <= max_x; ++x_px) {
                    vec2 pixel_center(static_cast<float>(x_px) + 0.5f, static_cast<float>(y_px) + 0.5f);// 픽셀 중심 좌표
                    vec3 bc = barycentric_coords(pixel_center, p_screen[0], p_screen[1], p_screen[2]);// 바리센트릭 좌표 계산

                    // 바리센트릭 좌표가 모두 0 이상이면 픽셀이 삼각형 내부에 있음
                    if (bc.x >= 0.0f && bc.y >= 0.0f && bc.z >= 0.0f) {
                        // 원근 보정 깊이 보간
                        float w_inv_interpolated = bc.x / w_clip[0] + bc.y / w_clip[1] + bc.z / w_clip[2];//보간된 1/w

                        float depth_ndc_interpolated = (bc.x * p_ndc[0].z / w_clip[0] +
                            bc.y * p_ndc[1].z / w_clip[1] +
                            bc.z * p_ndc[2].z / w_clip[2]) / w_inv_interpolated; // 보간된 깊이 값

                        // 보간된 NDC 깊이 값(-1~1)을 [0, 1] 범위로 변환 
                        float current_depth = depth_ndc_interpolated * 0.5f + 0.5f;

                        if (current_depth < ZBuffer[y_px * Width + x_px] && current_depth >= 0.0f && current_depth <= 1.0f) { // 깊이 검사 및 범위 확인 //현재픽셀이 이전픽셀보다 가까우면 갱신
                            ZBuffer[y_px * Width + x_px] = current_depth;
                            vec3 pos_numerator = bc.x * vert_pos_world[0] / w_clip[0] +
                                bc.y * vert_pos_world[1] / w_clip[1] +
                                bc.z * vert_pos_world[2] / w_clip[2]; // 보간된 색상
                            vec3 frag_pos_world_interpolated;
							frag_pos_world_interpolated = pos_numerator / w_inv_interpolated; // 보간된 색상
                            
							vec3 numerator = bc.x * vert_normal_world[0] / w_clip[0] +
								bc.y * vert_normal_world[1] / w_clip[1] +
								bc.z * vert_normal_world[2] / w_clip[2]; // 보간된 법선 벡터
							vec3 world_interpolated;
							world_interpolated = numerator / w_inv_interpolated; // 보간된 법선 벡터
							vec3 normal_world_interpolated = normalize(world_interpolated); // 법선 벡터 정규화

							vec3 Phong_color = calculate_blinn_phong_shading(
								frag_pos_world_interpolated,
								normal_world_interpolated,   
								eye_pos_world,              
								light_pos_world,             
								light_intensity,             
								ambient_intensity,           
								ka, kd, ks, shininess_p       
							);

                            vec3 color_before_gamma = Phong_color; // 보간된 색상 사용
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



//화면 표현
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
    ZBuffer.resize(static_cast<size_t>(Width) * Height);     // Z버퍼 크기 조절
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

    // 구 데이터 생성
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
        // 래스터화 렌더링 수행 (매 프레임 다시 계산)
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