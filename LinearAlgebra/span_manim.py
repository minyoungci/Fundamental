from manim import *

class VectorSpan(Scene):
    def construct(self):
        # 두 벡터 정의
        v1 = np.array([2, 1, 0])
        v2 = np.array([1, 2, 0])
        
        # 배경 그리드 추가
        grid = NumberPlane()
        self.add(grid)
        
        # 벡터 추가
        vec1 = Arrow(ORIGIN, v1, buff=0, color=RED)
        vec2 = Arrow(ORIGIN, v2, buff=0, color=BLUE)
        
        self.play(Create(vec1), Create(vec2))
        
        # 벡터에 레이블 추가
        label1 = MathTex(r"\vec{v}_1").next_to(vec1.get_end(), RIGHT)
        label2 = MathTex(r"\vec{v}_2").next_to(vec2.get_end(), UP)
        
        self.play(Write(label1), Write(label2))
        
        # Span 시각화
        span_area = Polygon(ORIGIN, v1, v1+v2, v2, color=GREEN, fill_opacity=0.5)
        self.play(Create(span_area))
        
        # 설명 텍스트 추가
        span_text = Text("Span of $\\vec{v}_1$ and $\\vec{v}_2$", font_size=24).to_edge(UP)
        self.play(Write(span_text))
        
        self.wait(2)

# Manim 명령어
# manim -pql script_name.py VectorSpan
