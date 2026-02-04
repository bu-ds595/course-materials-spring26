"""
Manim animation showing how a linear layer transforms space.
Shows a grid of points being transformed by a matrix W.
"""

from manim import *
import numpy as np

# Consistent color scheme
BLUE_VEC = ManimColor("#4A90D9")
RED_VEC = ManimColor("#E04040")
GRID_COLOR = ManimColor("#606060")
TEXT_COLOR = ManimColor("#1a1a1a")


class LinearLayerTransform(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        # Title in TeX
        title = MathTex(r"\text{Linear layer: } \mathbf{h} = \mathbf{W}\mathbf{x} + \mathbf{b}",
                       color=TEXT_COLOR, font_size=40)
        title.to_edge(UP, buff=0.5)

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

        # Create a grid of points with consistent color gradient
        points = []
        point_dots = VGroup()

        grid_range = np.linspace(-1.5, 1.5, 6)
        for x in grid_range:
            for y in grid_range:
                points.append(np.array([x, y]))
                # Color based on distance from origin
                t = (np.sqrt(x**2 + y**2)) / 2.2
                color = interpolate_color(BLUE_VEC, RED_VEC, min(t, 1))
                dot = Dot(axes.c2p(x, y), color=color, radius=0.07)
                point_dots.add(dot)

        # Basis vectors
        vec_x = Arrow(axes.c2p(0, 0), axes.c2p(1, 0), color=BLUE_VEC, buff=0,
                     stroke_width=5, max_tip_length_to_length_ratio=0.2)
        vec_y = Arrow(axes.c2p(0, 0), axes.c2p(0, 1), color=RED_VEC, buff=0,
                     stroke_width=5, max_tip_length_to_length_ratio=0.2)

        label_x = MathTex(r"\mathbf{e}_1", color=BLUE_VEC, font_size=30).next_to(vec_x.get_end(), DR, buff=0.1)
        label_y = MathTex(r"\mathbf{e}_2", color=RED_VEC, font_size=30).next_to(vec_y.get_end(), UL, buff=0.1)

        # Input label
        input_label = MathTex(r"\mathbf{x} \in \mathbb{R}^2", color=TEXT_COLOR, font_size=28)
        input_label.to_corner(DL, buff=0.5)

        # Show initial setup
        self.play(Write(title), run_time=0.8)
        self.play(Create(axes), run_time=0.6)
        self.play(
            FadeIn(point_dots, scale=0.5),
            GrowArrow(vec_x), GrowArrow(vec_y),
            Write(label_x), Write(label_y),
            FadeIn(input_label),
            run_time=1
        )
        self.wait(0.5)

        # Transformation matrix and bias
        W = np.array([[1.8, 0.5], [-0.3, 1.3]])
        b = np.array([0.5, -0.4])  # Bias vector

        # Show the matrix and bias
        matrix_tex = MathTex(
            r"\mathbf{W} = \begin{bmatrix} 1.8 & 0.5 \\ -0.3 & 1.3 \end{bmatrix}",
            color=TEXT_COLOR,
            font_size=26
        )
        bias_tex = MathTex(
            r"\mathbf{b} = \begin{bmatrix} 0.5 \\ -0.4 \end{bmatrix}",
            color=TEXT_COLOR,
            font_size=26
        )
        matrix_tex.to_corner(DR, buff=0.6)
        bias_tex.next_to(matrix_tex, DOWN, buff=0.2)

        self.play(Write(matrix_tex), Write(bias_tex))
        self.wait(0.3)

        # Create transformed objects
        def apply_transform(point):
            return W @ point

        new_point_dots = VGroup()
        for dot, pt in zip(point_dots, points):
            new_pt = apply_transform(pt)
            new_dot = Dot(axes.c2p(new_pt[0], new_pt[1]), color=dot.get_color(), radius=0.07)
            new_point_dots.add(new_dot)

        # Transform basis vectors
        new_e1 = apply_transform(np.array([1, 0]))
        new_e2 = apply_transform(np.array([0, 1]))

        new_vec_x = Arrow(axes.c2p(0, 0), axes.c2p(new_e1[0], new_e1[1]), color=BLUE_VEC,
                         buff=0, stroke_width=5, max_tip_length_to_length_ratio=0.15)
        new_vec_y = Arrow(axes.c2p(0, 0), axes.c2p(new_e2[0], new_e2[1]), color=RED_VEC,
                         buff=0, stroke_width=5, max_tip_length_to_length_ratio=0.15)

        new_label_x = MathTex(r"\mathbf{W}\mathbf{e}_1", color=BLUE_VEC, font_size=30).next_to(new_vec_x.get_end(), DR, buff=0.1)
        new_label_y = MathTex(r"\mathbf{W}\mathbf{e}_2", color=RED_VEC, font_size=30).next_to(new_vec_y.get_end(), UL, buff=0.1)

        # Output label
        output_label = MathTex(r"\mathbf{h} = \mathbf{W}\mathbf{x} \in \mathbb{R}^2", color=TEXT_COLOR, font_size=28)
        output_label.to_corner(DL, buff=0.5)

        # Step 1: Animate the Wx transformation
        step1_label = MathTex(r"\mathbf{W}\mathbf{x}", color=TEXT_COLOR, font_size=28)
        step1_label.to_corner(DL, buff=0.5)

        self.play(
            Transform(point_dots, new_point_dots),
            Transform(vec_x, new_vec_x),
            Transform(vec_y, new_vec_y),
            Transform(label_x, new_label_x),
            Transform(label_y, new_label_y),
            Transform(input_label, step1_label),
            run_time=2,
            rate_func=smooth
        )
        self.wait(0.5)

        # Step 2: Add bias (translation)
        # Create translated points
        bias_point_dots = VGroup()
        for dot, pt in zip(point_dots, points):
            new_pt = W @ pt + b
            new_dot = Dot(axes.c2p(new_pt[0], new_pt[1]), color=dot.get_color(), radius=0.07)
            bias_point_dots.add(new_dot)

        # Translate vectors too
        bias_vec_x = Arrow(axes.c2p(b[0], b[1]), axes.c2p(new_e1[0] + b[0], new_e1[1] + b[1]),
                          color=BLUE_VEC, buff=0, stroke_width=5, max_tip_length_to_length_ratio=0.15)
        bias_vec_y = Arrow(axes.c2p(b[0], b[1]), axes.c2p(new_e2[0] + b[0], new_e2[1] + b[1]),
                          color=RED_VEC, buff=0, stroke_width=5, max_tip_length_to_length_ratio=0.15)

        bias_label_x = MathTex(r"\mathbf{W}\mathbf{e}_1", color=BLUE_VEC, font_size=30).next_to(bias_vec_x.get_end(), DR, buff=0.1)
        bias_label_y = MathTex(r"\mathbf{W}\mathbf{e}_2", color=RED_VEC, font_size=30).next_to(bias_vec_y.get_end(), UL, buff=0.1)

        # Final output label
        output_label = MathTex(r"\mathbf{h} = \mathbf{W}\mathbf{x} + \mathbf{b}", color=TEXT_COLOR, font_size=28)
        output_label.to_corner(DL, buff=0.5)

        # Show + b text briefly
        plus_b = MathTex(r"+ \mathbf{b}", color=TEXT_COLOR, font_size=36)
        plus_b.move_to(axes.c2p(0, 0))

        self.play(FadeIn(plus_b, scale=1.5), run_time=0.5)
        self.play(
            Transform(point_dots, bias_point_dots),
            Transform(vec_x, bias_vec_x),
            Transform(vec_y, bias_vec_y),
            Transform(label_x, bias_label_x),
            Transform(label_y, bias_label_y),
            Transform(input_label, output_label),
            FadeOut(plus_b),
            run_time=1.5,
            rate_func=smooth
        )
        self.wait(0.8)

        # Explanatory text in TeX
        explain = MathTex(r"\mathbf{W} \text{ rotates/stretches, } \mathbf{b} \text{ translates}",
                         font_size=26, color=GRID_COLOR)
        explain.next_to(axes, DOWN, buff=0.5)
        self.play(FadeIn(explain))
        self.wait(2)


if __name__ == "__main__":
    # Render with: manim -qm --format=gif linear_transform_anim.py LinearLayerTransform
    pass
