About
===
Phong Shadingx
---
How to Use
===
1. Download zip piles.  
   
![download zip](https://github.com/user-attachments/assets/3e76e9d2-5325-42a3-ba52-2bb3064c0a58)

2. Unzip the folder  
3. open "OpenglViewer.sln"  
![leanch](https://github.com/user-attachments/assets/1ed43ef3-d812-4b75-809d-fe1077eabf9b)
---
Result of assignmet6-3

![스크린샷 2025-05-08 162804](https://github.com/user-attachments/assets/4fb7d976-5b9a-4d1a-8d94-a68f3d9341df)

---
Explanation
---
The key to Phong shading is to interpolate the normal vector for each pixel, and to use this interpolated normal vector and interpolated position to do light calculations directly from each pixel

![image](https://github.com/user-attachments/assets/e6bca9b2-a16d-491d-a016-a65f47aa1997)

We're going to do shading calculations for each pixel, so there's no need for a vertex color calculation

![image](https://github.com/user-attachments/assets/0c51d3f9-2122-44e3-95ec-6051a64fb8b6)

In the same way we interpolated colors in Q2, we calculate frag_pos_world_interpolated using the center-of-gravity coordinates of the current pixel and the clip space w value.
The three vertices are perspectively corrected and normalized.

Using two interpolated values to call a Phong-Shading function to obtain a phong color

![image](https://github.com/user-attachments/assets/71b94aa4-8135-4932-994f-8a83581e908c)




