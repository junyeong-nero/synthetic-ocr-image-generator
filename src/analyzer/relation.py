import argparse
from datasets import load_dataset
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import chi2_contingency


def relation_analysis_interactive(dataset_id, column_a, column_b, split="train"):
    """
    Hugging Face 데이터셋의 두 열(column_a, column_b) 간의 관계를 분석하고
    Plotly를 사용하여 인터랙티브하게 시각화합니다.
    (Jupyter Notebook, Google Colab과 같은 환경에 최적화되어 있습니다.)

    Args:
        dataset_id (str): Hugging Face Hub의 데이터셋 ID.
        column_a (str): 분석할 첫 번째 열의 이름.
        column_b (str): 분석할 두 번째 열의 이름.
        split (str, optional): 사용할 데이터셋 스플릿. 기본값은 "train".
    """
    # 1. 데이터셋 로드 및 Pandas DataFrame으로 변환
    try:
        print(f"'{dataset_id}' 데이터셋의 '{split}' 스플릿을 로드합니다...")
        dataset = load_dataset(dataset_id, split=split)
        df = pd.DataFrame(dataset)
        print("데이터셋 로드 완료.")
        print("\n데이터셋 정보:")
        print(df.info())
        print("\n데이터셋 샘플:")
        print(df.head())
    except Exception as e:
        print(f"데이터셋 로드에 실패했습니다: {e}")
        return

    # 2. 분석할 열이 데이터프레임에 있는지 확인
    if column_a not in df.columns or column_b not in df.columns:
        print(f"오류: '{column_a}' 또는 '{column_b}' 열을 찾을 수 없습니다.")
        print(f"사용 가능한 열: {df.columns.tolist()}")
        return

    # 3. 데이터 유형 확인 (숫자형, 범주형)
    is_a_numeric = pd.api.types.is_numeric_dtype(df[column_a])
    is_b_numeric = pd.api.types.is_numeric_dtype(df[column_b])

    # 4. 데이터 유형 조합에 따라 분석 및 시각화
    # 경우 1: 두 열 모두 숫자형일 경우
    if is_a_numeric and is_b_numeric:
        print(f"\n--- 숫자형 vs 숫자형 분석: '{column_a}' vs '{column_b}' ---")
        correlation = df[column_a].corr(df[column_b])
        print(f"\n피어슨 상관계수: {correlation:.4f}")

        fig_scatter = px.scatter(
            df,
            x=column_a,
            y=column_b,
            title=f"<b>산점도: {column_a} vs {column_b}</b><br>상관계수: {correlation:.2f}",
            trendline="ols",
            trendline_color_override="red",
        )
        fig_scatter.show()

        corr_matrix = df[[column_a, column_b]].corr()
        fig_heatmap = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title=f"<b>상관관계 히트맵: {column_a} vs {column_b}</b>",
        )
        fig_heatmap.show()

    # 경우 2: 두 열 모두 범주형일 경우
    elif not is_a_numeric and not is_b_numeric:
        print(f"\n--- 범주형 vs 범주형 분석: '{column_a}' vs '{column_b}' ---")
        contingency_table = pd.crosstab(df[column_a], df[column_b])
        print("\n교차표:")
        print(contingency_table)

        chi2, p, _, _ = chi2_contingency(contingency_table)
        print(f"\n카이제곱 통계량: {chi2:.4f}, p-value: {p:.4f}")
        if p < 0.05:
            print("결론: 두 변수는 통계적으로 유의미한 관계가 있습니다.")
        else:
            print("결론: 두 변수는 독립적일 가능성이 높습니다.")

        fig_heatmap_cat = px.imshow(
            contingency_table,
            text_auto=True,
            aspect="auto",
            title=f"<b>교차표 히트맵: {column_a} vs {column_b}</b>",
        )
        fig_heatmap_cat.show()

        fig_bar = px.histogram(
            df,
            x=column_a,
            color=column_b,
            barmode="group",
            title=f"<b>그룹 막대 그래프: {column_b}에 따른 {column_a} 분포</b>",
        )
        fig_bar.show()

    # 경우 3: 하나는 숫자형, 다른 하나는 범주형일 경우
    else:
        numeric_col, categoric_col = (
            (column_a, column_b) if is_a_numeric else (column_b, column_a)
        )
        print(f"\n--- 숫자형 vs 범주형 분석: '{numeric_col}' vs '{categoric_col}' ---")
        print(f"\n'{categoric_col}'에 따른 '{numeric_col}'의 기술 통계량:")
        print(df.groupby(categoric_col)[numeric_col].describe())

        fig_box = px.box(
            df,
            x=categoric_col,
            y=numeric_col,
            color=categoric_col,
            title=f"<b>박스플롯: {categoric_col}에 따른 {numeric_col} 분포</b>",
        )
        fig_box.show()

        fig_violin = px.violin(
            df,
            x=categoric_col,
            y=numeric_col,
            color=categoric_col,
            box=True,
            points="all",
            title=f"<b>바이올린플롯: {categoric_col}에 따른 {numeric_col} 분포</b>",
        )
        fig_violin.show()


def main():
    """
    커맨드 라인 인수를 파싱하여 관계 분석 함수를 실행합니다.
    """
    parser = argparse.ArgumentParser(
        description="Hugging Face 데이터셋의 두 열 간의 관계를 분석하고 인터랙티브하게 시각화합니다.",
        formatter_class=argparse.RawTextHelpFormatter,  # 도움말 포맷을 예쁘게 보여주기 위함
    )

    parser.add_argument(
        "dataset_id",
        type=str,
        help="Hugging Face Hub의 데이터셋 ID (예: scikit-learn/iris)",
    )
    parser.add_argument(
        "column_a", type=str, help="분석할 첫 번째 열의 이름 (예: sepal.length)"
    )
    parser.add_argument(
        "column_b", type=str, help="분석할 두 번째 열의 이름 (예: petal.length)"
    )
    parser.add_argument(
        "--split",
        "-s",
        type=str,
        default="train",
        help="사용할 데이터셋 스플릿 (기본값: train)",
    )

    args = parser.parse_args()

    # 파싱된 인수로 메인 함수 실행
    relation_analysis_interactive(
        dataset_id=args.dataset_id,
        column_a=args.column_a,
        column_b=args.column_b,
        split=args.split,
    )


if __name__ == "__main__":
    main()
