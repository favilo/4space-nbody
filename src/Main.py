
import sys

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL import platform, GLX, WGL

import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.elementwise import ElementwiseKernel

import numpy

from FourSpace import *

class NBody:
    ambient_light  = [0.5, 0.5, 0.0, 1]
    diffuse_light  = [1, 1, 1, 0.5]
    specular_light = [1, 1, 1, 0.8]
    light_position = [0, 0, 2, 1]

    eye_pos = ( 0, 0, 6, 0)
    cen_pos = ( 0, 0, 0, 0)

    def __init__(self):
        plats = cl.get_platforms()
        ctx_props = cl.context_properties

        self.props = [(ctx_props.PLATFORM, plats[0]),
                 (ctx_props.GL_CONTEXT_KHR, platform.GetCurrentContext())]

        if sys.platform == "linux2":
            self.props.append((ctx_props.GLX_DISPLAY_KHR,
                            GLX.glXGetCurrentDisplay()))
        elif sys.platform == "win32":
            self.props.append((ctx_props.WGL_HDC_KHR,
                            WGL.wglGetCurrentDC()))
        self.ctx = cl.Context(properties=self.props)

        self.cross4 = ElementwiseKernel(self.ctx,
                "__global const float4 *u, "
                "__global const float4 *v, "
                "__global const float4 *w, "
                "__global       float4 *r",
                "r[i] = cross4(u[i],v[i],w[i])",
                "cross4_final", preamble=cross4_preamble)

        self.distance2 = ElementwiseKernel(self.ctx,
                "__global const float4 *a, "
                "__global const float4 *b, "
                "__global       float4 *d",
                "d[i] = distance2(a[i],b[i])",
                "distance_final", preamble=distance_preamble)
        self.place_hyperspheres()

    def test(self):
        a = numpy.random.randn(4,4).astype(numpy.float32)
        b = numpy.random.randn(4,4).astype(numpy.float32)
        c = numpy.random.randn(4,4).astype(numpy.float32)

        a_gpu = cl_array.to_device(self.ctx, queue, a)
        b_gpu = cl_array.to_device(self.ctx, queue, b)
        c_gpu = cl_array.to_device(self.ctx, queue, c)

        dest_gpu = cl_array.empty_like(a_gpu)

    def gl_init(self):
        glShadeModel(GL_SMOOTH)
        glClearColor(0, 0, 0, 0.5)

        glClearDepth(1.0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)

        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)

        glLightfv(GL_LIGHT1, GL_AMBIENT, self.ambient_light)
        glLightfv(GL_LIGHT1, GL_DIFFUSE, self.diffuse_light)
        glLightfv(GL_LIGHT1, GL_SPECULAR, self.specular_light)
        glLightfv(GL_LIGHT1, GL_POSITION, self.light_position)

        glEnable(GL_LIGHT1)
        glEnable(GL_LIGHTING)

        self.build_list()

    def build_list(self):
        quadric = gluNewQuadric()
        gluQuadricNormals(quadric, GLU_SMOOTH)
        gluQuadricOrientation(quadric, GLU_OUTSIDE)
        gluQuadricTexture(quadric, True)

        self.sphere = glGenLists(1)
        glNewList(self.sphere, GL_COMPILE)
        gluSphere(quadric, 1, 64, 64)
        glEndList()

    def place_hyperspheres(self):
        self.spheres = [((0,0,-1,0),(0.3,0.8,1))]
        self.spheres.append(((0,2,0,0.5),(1,0,0.33)))
        self.spheres.append(((0,-2,0,-0.3),(0.5,0,1)))
        self.spheres.append(((2,0,0,0.5),(0.66,0.5,1)))

def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    (x,y,z,w) = nBody.eye_pos
    (x0,y0,z0,w0) = nBody.cen_pos
    gluLookAt(x,y,z,
              x0,y0,z0,
              0,1,0)

    for ((x,y,z,w),(r,g,b)) in nBody.spheres:
        glPushMatrix()
        glColor(r,g,b)
        glTranslate(x,y,z)
        glScale(1-w,1-w,1-w)
        glCallList(nBody.sphere)
        glPopMatrix()
    glutSwapBuffers()


def reshape(w, h):
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    if ( h == 0):
        gluPerspective( 80, w, 1, 5000)
    else:
        gluPerspective( 80, float(w)/float(h), 1, 5000)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def keyboard(key, x, y):
    sys.exit(0)

def mouse(button, state, x, y):
    pass

def main():
    global nBody

    glutInit(sys.argv)

    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(500, 500)
    glutInitWindowPosition(0,0)
    glutCreateWindow('4-Space nBody Simulation')
    nBody = NBody()
    nBody.gl_init()

    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutMouseFunc(mouse)
    glutIdleFunc(display)

    glutMainLoop()

if __name__ == "__main__":
    main()
