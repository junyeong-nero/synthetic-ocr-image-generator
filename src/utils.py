import json


def read_json(file_path):
    """
    JSON 파일을 읽어 파이썬 객체로 반환합니다.

    Args:
        file_path (str): 읽을 JSON 파일의 경로.

    Returns:
        dict or list: JSON 파일의 내용을 담은 파이썬 객체.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"오류: 파일 '{file_path}'를 찾을 수 없습니다.")
        return None
    except json.JSONDecodeError:
        print(f"오류: '{file_path}' 파일이 올바른 JSON 형식이 아닙니다.")
        return None


import json


def save_json(data, file_path):
    """
    파이썬 객체를 JSON 파일로 저장합니다.

    Args:
        data (dict or list): 저장할 파이썬 객체.
        file_path (str): 저장할 JSON 파일의 경로.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"데이터가 '{file_path}' 파일에 성공적으로 저장되었습니다.")
    except Exception as e:
        print(f"파일 저장 중 오류가 발생했습니다: {e}")


def read_txt(file_path):
    """
    텍스트 파일의 모든 줄을 읽어 리스트로 반환합니다.

    Args:
        file_path (str): 읽을 텍스트 파일의 경로.

    Returns:
        list: 파일의 각 줄을 요소로 하는 문자열 리스트.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        # 각 줄의 끝에 있는 개행 문자(\n) 제거
        return text
    except FileNotFoundError:
        print(f"오류: 파일 '{file_path}'를 찾을 수 없습니다.")
        return None


def save_txt(lines, file_path):
    """
    문자열 리스트를 텍스트 파일에 씁니다. 각 요소는 한 줄에 해당합니다.

    Args:
        lines (list): 파일에 쓸 문자열의 리스트.
        file_path (str): 저장할 텍스트 파일의 경로.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        print(f"데이터가 '{file_path}' 파일에 성공적으로 저장되었습니다.")
    except Exception as e:
        print(f"파일 저장 중 오류가 발생했습니다: {e}")
