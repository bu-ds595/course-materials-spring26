"""
Manim animation showing how nonlinearity bends space.
Shows: linear transform, then ReLU/tanh applied.
"""

from manim import *
import numpy as np

# Consistent color scheme
BLUE_VEC = ManimColor("#4A90D9")
RED_VEC = ManimColor("#E04040")
GRID_COLOR = ManimColor("#606060")
TEXT_COLOR = ManimColor("#1a1a1a")
GREEN_ACT = ManimColor("#50C878")


class NonlinearityTransform(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        # Title
        title = MathTex(r"\text{Adding nonlinearity: } \mathbf{h} = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b})",
                       color=TEXT_COLOR, font_size=36)
        title.to_edge(UP, buff=0.4)

        # Create coordinate axes
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=5,
            y_length=5,
            axis_config={"color": GRID_COLOR, "stroke_width": 1.5, "include_ticks": False},
            tips=False,
        )
        axes.shift(DOWN * 0.3)

        # Create a grid of points
        points = []
        point_dots = VGroup()

        grid_range = np.linspace(-1.2, 1.2, 7)
        for x in grid_range:
            for y in grid_range:
                points.append(np.array([x, y]))
                t = (np.sqrt(x**2 + y**2)) / 1.8
                color = interpolate_color(BLUE_VEC, RED_VEC, min(t, 1))
                dot = Dot(axes.c2p(x, y), color=color, radius=0.06)
                point_dots.add(dot)

        # Input label
        input_label = MathTex(r"\mathbf{x}", color=TEXT_COLOR, font_size=28)
        input_label.to_corner(DL, buff=0.5)

        # Show initial setup
        self.play(Write(title), run_time=0.8)
        self.play(Create(axes), run_time=0.5)
        self.play(
            FadeIn(point_dots, scale=0.5),
            FadeIn(input_label),
            run_time=0.8
        )
        self.wait(0.3)

        # Linear transformation
        W = np.array([[1.5, 0.4], [-0.3, 1.2]])
        b = np.array([0.3, -0.2])

        # Show equation
        eq_linear = MathTex(r"\mathbf{W}\mathbf{x} + \mathbf{b}", color=TEXT_COLOR, font_size=28)
        eq_linear.to_corner(DL, buff=0.5)

        # Apply linear transform
        linear_points = []
        linear_dots = VGroup()
        for pt, dot in zip(points, point_dots):
            new_pt = W @ pt + b
            linear_points.append(new_pt)
            new_dot = Dot(axes.c2p(new_pt[0], new_pt[1]), color=dot.get_color(), radius=0.06)
            linear_dots.add(new_dot)

        self.play(
            Transform(point_dots, linear_dots),
            Transform(input_label, eq_linear),
            run_time=1.5,
            rate_func=smooth
        )
        self.wait(0.5)

        # Now apply nonlinearity (tanh - smooth and visible effect)
        activation_label = MathTex(r"\sigma = \tanh", color=GREEN_ACT, font_size=26)
        activation_label.to_corner(DR, buff=0.5)

        eq_nonlinear = MathTex(r"\sigma(\mathbf{W}\mathbf{x} + \mathbf{b})", color=TEXT_COLOR, font_size=28)
        eq_nonlinear.to_corner(DL, buff=0.5)

        self.play(Write(activation_label))

        # Apply tanh elementwise
        nonlinear_dots = VGroup()
        for pt, dot in zip(linear_points, point_dots):
            # Apply tanh to each coordinate
            new_pt = np.tanh(pt * 1.5)  # Scale up to make effect more visible
            new_dot = Dot(axes.c2p(new_pt[0], new_pt[1]), color=dot.get_color(), radius=0.06)
            nonlinear_dots.add(new_dot)

        # Show "applying σ" text
        sigma_text = MathTex(r"\sigma(\cdot)", color=GREEN_ACT, font_size=40)
        sigma_text.move_to(axes.c2p(0, 0))

        self.play(FadeIn(sigma_text, scale=1.5), run_time=0.4)
        self.play(
            Transform(point_dots, nonlinear_dots),
            Transform(input_label, eq_nonlinear),
            FadeOut(sigma_text),
            run_time=2,
            rate_func=smooth
        )
        self.wait(0.5)

        # Explanation
        explain = MathTex(r"\text{Nonlinearity bends and compresses space}",
                         font_size=24, color=GRID_COLOR)
        explain.next_to(axes, DOWN, buff=0.4)
        self.play(FadeIn(explain))
        self.wait(2)


class ReLUTransform(Scene):
    """Show ReLU specifically - more dramatic effect"""
    def construct(self):
        self.camera.background_color = WHITE

        # Title
        title = MathTex(r"\text{ReLU: } \sigma(z) = \max(0, z)",
                       color=TEXT_COLOR, font_size=36)
        title.to_edge(UP, buff=0.4)

        # Create axes
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=5,
            y_length=5,
            axis_config={"color": GRID_COLOR, "stroke_width": 1.5, "include_ticks": False},
            tips=False,
        )
        axes.shift(DOWN * 0.3)

        # Create grid
        points = []
        point_dots = VGroup()

        grid_range = np.linspace(-1.5, 1.5, 8)
        for x in grid_range:
            for y in grid_range:
                points.append(np.array([x, y]))
                t = (np.sqrt(x**2 + y**2)) / 2.2
                color = interpolate_color(BLUE_VEC, RED_VEC, min(t, 1))
                dot = Dot(axes.c2p(x, y), color=color, radius=0.055)
                point_dots.add(dot)

        input_label = MathTex(r"\mathbf{x}", color=TEXT_COLOR, font_size=28)
        input_label.to_corner(DL, buff=0.5)

        self.play(Write(title), Create(axes), run_time=0.8)
        self.play(FadeIn(point_dots, scale=0.5), FadeIn(input_label), run_time=0.8)
        self.wait(0.3)

        # Linear transform first
        W = np.array([[1.3, 0.5], [0.2, 1.1]])
        b = np.array([-0.3, -0.3])

        linear_points = []
        linear_dots = VGroup()
        for pt, dot in zip(points, point_dots):
            new_pt = W @ pt + b
            linear_points.append(new_pt)
            new_dot = Dot(axes.c2p(new_pt[0], new_pt[1]), color=dot.get_color(), radius=0.055)
            linear_dots.add(new_dot)

        eq_linear = MathTex(r"\mathbf{W}\mathbf{x} + \mathbf{b}", color=TEXT_COLOR, font_size=28)
        eq_linear.to_corner(DL, buff=0.5)

        self.play(
            Transform(point_dots, linear_dots),
            Transform(input_label, eq_linear),
            run_time=1.5
        )
        self.wait(0.4)

        # Apply ReLU
        relu_dots = VGroup()
        for pt, dot in zip(linear_points, point_dots):
            new_pt = np.maximum(0, pt)  # ReLU
            new_dot = Dot(axes.c2p(new_pt[0], new_pt[1]), color=dot.get_color(), radius=0.055)
            relu_dots.add(new_dot)

        eq_relu = MathTex(r"\text{ReLU}(\mathbf{W}\mathbf{x} + \mathbf{b})", color=TEXT_COLOR, font_size=28)
        eq_relu.to_corner(DL, buff=0.5)

        relu_text = MathTex(r"\text{ReLU}", color=GREEN_ACT, font_size=40)
        relu_text.move_to(axes.c2p(0, 0))

        self.play(FadeIn(relu_text, scale=1.5), run_time=0.4)
        self.play(
            Transform(point_dots, relu_dots),
            Transform(input_label, eq_relu),
            FadeOut(relu_text),
            run_time=2,
            rate_func=smooth
        )
        self.wait(0.5)

        # Explanation
        explain = MathTex(r"\text{ReLU folds space at zero}",
                         font_size=24, color=GRID_COLOR)
        explain.next_to(axes, DOWN, buff=0.4)
        self.play(FadeIn(explain))
        self.wait(2)


if __name__ == "__main__":
    pass
