# my Software Renderer

- About

  Now I'm spending some time building a soft-renderer. It won't be so noble though, but it's meaningful to me.  I'm try to build it by C++ from the scratch, using as few libraries as possible. Definitely APIs like openGL would NOT be used in this proj.

- Reference

  I refer the known soft-renderer tutorials, [tinyrenderer](https://github.com/ssloy/tinyrenderer/wiki). However, my codes are quite different from the tutorials contents. I code mainly with the help of the [tigerbook](https://www.amazon.com/Fundamentals-Computer-Graphics-Steve-Marschner/dp/0367505037). After master the knowledge in the book, I try to coding myself. It's pretty funny.

- Libraries

  As the original tutorials states, the external libraries included in this project is only Tgaimage. Of course, some basic libraries are also included, such as Geometry.

- Functions already finished

  - render a raw model
  - orthogonal/perspective projection
  - hidden face removal (z-buffer)

  - Gouraud shading

  - Phong Shading
  - self-shadowing (hard)
  - output depth map

- Functions expected

  - ray tracing
  - anti-aliasing
  - soft shadowing
  - clipping
  - texturing
  - more and more smart shaders
  - ......

- Simply have a look

  I'm sorry I still not implement the GUI and export this project as an .exe file. Now I just want to focus on the graphics stuff. If you want to have a check for the output, please open .sln file by VisualStudio and simply run the project, then open the output.tga (sorry again, for not using popular image format) in the "myrenderer" folder.

  If you want to have a check on the renderer code, please open mygl.h and main.cpp files. These 2 are the main codes, and others are just libraries or sth dependence.

- Others

  I start this plan in the early April 2023, and it's still very naive and immature. I plan to continuously shape the codes and improve this project as possible as I can.
