import cv2
from PIL import Image
import os
import argparse

# --- 設定 ---
# 顔検出用の分類器ファイルのパス (スクリプトと同じフォルダにある場合)
# 環境に合わせてパスを調整してください
CASCADE_PATH = 'haarcascade_frontalface_default.xml'
OUTPUT_SIZE = (80, 80)  # 出力画像のピクセルサイズ (幅, 高さ)
OUTPUT_FORMAT = 'png'  # 出力画像のフォーマット (JPEGなども可)
FACE_MARGIN_RATIO = 2.0 # 顔検出領域に対するトリミング領域の倍率 (1.0だと顔ギリギリ)

# --- 関数定義 ---

def detect_face(image_path, cascade_path):
    """
    画像から顔を検出する関数
    Args:
        image_path (str): 入力画像のパス
        cascade_path (str): 顔検出用分類器(xml)のパス
    Returns:
        tuple or None: 検出された顔の領域 (x, y, w, h) のリスト。見つからない場合はNone。
                       一番大きく検出された顔の情報を返すように変更。
    """
    if not os.path.exists(cascade_path):
        print(f"エラー: 顔検出ファイルが見つかりません: {cascade_path}")
        return None
    if not os.path.exists(image_path):
        print(f"エラー: 入力画像ファイルが見つかりません: {image_path}")
        return None

    # OpenCVで画像を読み込み (日本語パス対応のため numpy 経由で読み込む)
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"エラー: 画像ファイルを読み込めませんでした: {image_path}")
            return None
        # グレースケールに変換
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print(f"エラー: OpenCVでの画像読み込み中にエラーが発生しました: {e}")
        return None

    # 顔検出器をロード
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print(f"エラー: 顔検出器のロードに失敗しました: {cascade_path}")
        return None

    # 顔を検出
    # scaleFactor: 画像スケール縮小量 (大きいほど検出は早いが精度が落ちる)
    # minNeighbors: 物体候補となる矩形は，最低でも指定した数だけの近傍矩形を含む必要がある
    # minSize: 物体が取り得る最小サイズ．これより小さい物体は無視される
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print(f"情報: {os.path.basename(image_path)} で顔が検出されませんでした。")
        return None
    else:
        # 検出された顔の中から一番大きいものを選択 (面積で比較)
        largest_face = max(faces, key=lambda item: item[2] * item[3])
        print(f"情報: {os.path.basename(image_path)} で顔を検出しました。領域: {largest_face}")
        return largest_face # (x, y, w, h)

def crop_to_square_around_face(img_pil, face_rect, margin_ratio=1.4):
    """
    検出された顔を中心に正方形にトリミングする関数
    Args:
        img_pil (PIL.Image): Pillowで読み込んだ画像オブジェクト
        face_rect (tuple): 顔の領域 (x, y, w, h)
        margin_ratio (float): 顔のサイズに対するトリミング領域の倍率
    Returns:
        PIL.Image: トリミングされた画像オブジェクト
    """
    x, y, w, h = face_rect
    img_width, img_height = img_pil.size

    # 顔の中心座標
    center_x = x + w // 2
    center_y = y + h // 2

    # トリミングする正方形の一辺の長さ (顔の大きい方の辺 * マージン倍率)
    crop_size = int(max(w, h) * margin_ratio)

    # トリミング領域の計算 (中心基準)
    x1 = center_x - crop_size // 2
    y1 = center_y - crop_size // 2
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    # 画像の境界からはみ出ないように調整
    # 左上座標の調整
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    # 右下座標の調整 (はみ出た分を考慮してサイズを再計算)
    if x2 > img_width:
        x2 = img_width
    if y2 > img_height:
        y2 = img_height

    # はみ出し調整後にサイズが変わった可能性があるので、正方形になるように再調整
    # (小さい方の辺に合わせて大きい方の辺を中央寄せで縮める)
    current_w = x2 - x1
    current_h = y2 - y1

    if current_w > current_h: # 横長の場合
        diff = current_w - current_h
        x1 += diff // 2
        x2 = x1 + current_h # 正方形にする
    elif current_h > current_w: # 縦長の場合
        diff = current_h - current_w
        y1 += diff // 2
        y2 = y1 + current_w # 正方形にする

    # 最終的なトリミング領域
    # Pillow の crop は (left, upper, right, lower)
    box = (x1, y1, x2, y2)
    print(f"情報: トリミング領域を計算しました: {box}")

    return img_pil.crop(box)

def crop_center_square(img_pil):
    """
    画像の中心を正方形にトリミングする関数 (顔検出失敗時のフォールバック)
    Args:
        img_pil (PIL.Image): Pillowで読み込んだ画像オブジェクト
    Returns:
        PIL.Image: トリミングされた画像オブジェクト
    """
    img_width, img_height = img_pil.size
    crop_size = min(img_width, img_height)

    left = (img_width - crop_size) // 2
    top = (img_height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size

    box = (left, top, right, bottom)
    print(f"情報: 顔検出失敗のため、中央をトリミングします。領域: {box}")
    return img_pil.crop(box)

def process_image(input_path, output_path, cascade_path, output_size, output_format, face_margin_ratio):
    """
    画像処理のメイン関数
    """
    try:
        # Pillowで画像を読み込み (こちらで加工・保存を行う)
        img_pil = Image.open(input_path)
        # RGBA (透過情報付き) の場合、JPEG保存時にエラーになるためRGBに変換
        if img_pil.mode == 'RGBA' and output_format.upper() == 'JPEG':
            print("情報: RGBA画像をRGBに変換します (JPEG保存のため)")
            img_pil = img_pil.convert('RGB')
        elif img_pil.mode not in ['RGB', 'L']: # Lはグレースケール
             # 対応していないモードの場合はRGBに変換してみる
             print(f"情報: 画像モード {img_pil.mode} をRGBに変換します")
             try:
                 img_pil = img_pil.convert('RGB')
             except Exception as convert_e:
                 print(f"エラー: モード {img_pil.mode} からRGBへの変換に失敗しました: {convert_e}")
                 return False

    except FileNotFoundError:
        print(f"エラー: 入力画像ファイルが見つかりません: {input_path}")
        return False
    except Exception as e:
        print(f"エラー: Pillowでの画像読み込み/変換中にエラーが発生しました: {e}")
        return False

    # 顔を検出
    face_rect = detect_face(input_path, cascade_path)

    # トリミング処理
    # ★★★ 修正箇所 ★★★
    if face_rect is not None:  # face_rect が None でないかを確認
        cropped_img = crop_to_square_around_face(img_pil, face_rect, face_margin_ratio)
    else:
        # 顔が見つからなかった場合は中央をトリミング
        cropped_img = crop_center_square(img_pil)

    # リサイズ (高品質なアンチエイリアスを使用)
    try:
        # Pillow 9.1.0以降では Image.Resampling.LANCZOS を推奨
        resample_filter = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
        resized_img = cropped_img.resize(output_size, resample=resample_filter)
    except Exception as e:
        print(f"エラー: リサイズ中にエラーが発生しました: {e}")
        return False

    # 出力ディレクトリが存在しない場合は作成
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"情報: 出力ディレクトリを作成しました: {output_dir}")
        except OSError as e:
            print(f"エラー: 出力ディレクトリの作成に失敗しました: {e}")
            return False

    # 保存
    try:
        # quality は JPEG の場合に有効 (1-95, デフォルト 75)
        # optimize は JPEG, PNG で有効 (ファイルサイズを最適化)
        if output_format.upper() == 'JPEG':
            resized_img.save(output_path, format=output_format, quality=90, optimize=True)
        else:
            resized_img.save(output_path, format=output_format, optimize=True)
        print(f"成功: 画像を編集し、 {output_path} に {output_format} 形式で保存しました。")
        return True
    except Exception as e:
        print(f"エラー: 画像の保存中にエラーが発生しました ({output_path}): {e}")
        # 保存しようとしたファイルが残っている場合があるので削除を試みる
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass # 削除できなくても無視
        return False

# --- メイン処理 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='エージェント画像を自動編集します (顔検出->正方形トリミング->リサイズ)')
    parser.add_argument('input', help='入力画像ファイルまたは画像ファイルが含まれるディレクトリのパス')
    parser.add_argument('-o', '--output', help='出力ファイルまたは出力ディレクトリのパス (省略時は入力と同じ場所に"_edited"をつけて保存)')
    parser.add_argument('-c', '--cascade', default=CASCADE_PATH, help=f'顔検出用分類器(haarcascade)のXMLファイルのパス (デフォルト: {CASCADE_PATH})')
    parser.add_argument('-s', '--size', default=f"{OUTPUT_SIZE[0]}x{OUTPUT_SIZE[1]}", help=f'出力画像のサイズ (幅x高さ) (デフォルト: {OUTPUT_SIZE[0]}x{OUTPUT_SIZE[1]})')
    parser.add_argument('-f', '--format', default=OUTPUT_FORMAT, choices=['PNG', 'JPEG', 'GIF', 'BMP', 'TIFF'], help=f'出力画像のフォーマット (デフォルト: {OUTPUT_FORMAT})')
    parser.add_argument('-m', '--margin', type=float, default=FACE_MARGIN_RATIO, help=f'顔検出領域に対するトリミング領域の倍率 (デフォルト: {FACE_MARGIN_RATIO})')

    args = parser.parse_args()

    # 出力サイズの解析
    try:
        width, height = map(int, args.size.split('x'))
        output_size = (width, height)
        if width != height:
            print("警告: 出力サイズが正方形ではありません。正方形トリミング後に指定サイズにリサイズされます。")
    except ValueError:
        print(f"エラー: サイズ指定の形式が不正です ('幅x高さ'形式で指定してください): {args.size}")
        exit(1)

    input_path = args.input
    output_path_arg = args.output

    # 入力がディレクトリかファイルか判定
    if os.path.isdir(input_path):
        print(f"情報: ディレクトリ内の画像を処理します: {input_path}")
        if output_path_arg and not os.path.isdir(output_path_arg):
             # 出力先が指定され、かつそれがディレクトリでない場合はエラー
             print(f"エラー: 入力がディレクトリの場合、出力先もディレクトリとして指定するか、省略してください。: {output_path_arg}")
             exit(1)
        elif output_path_arg:
            # 出力ディレクトリが指定されている場合
            output_dir = output_path_arg
        else:
            # 出力ディレクトリが省略されている場合、入力ディレクトリ内に出力用サブディレクトリを作成
             output_dir = os.path.join(input_path, "edited_images")

        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                print(f"情報: 出力ディレクトリを作成しました: {output_dir}")
            except OSError as e:
                 print(f"エラー: 出力ディレクトリの作成に失敗しました: {e}")
                 exit(1)

        processed_count = 0
        error_count = 0
        skipped_count = 0
        for filename in os.listdir(input_path):
            input_file = os.path.join(input_path, filename)
            # 画像ファイルかどうかを簡易的に判定 (拡張子で)
            if os.path.isfile(input_file) and filename.lower().endswith(('.webp', '.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.WEBP', '.PNG', '.JPG', '.JPEG', '.BMP', '.GIF', '.TIFF', '.jfif', '.JPE', '.svg', '.SVG', '.heic', '.HEIC', '.avif', '.AVIF')):
                base, ext = os.path.splitext(filename)
                output_filename = f"{base}_edited.{args.format.lower()}"
                output_file = os.path.join(output_dir, output_filename)
                print(f"\n--- 処理開始: {filename} ---")
                if process_image(input_file, output_file, args.cascade, output_size, args.format, args.margin):
                    processed_count += 1
                else:
                    error_count += 1
            elif os.path.isfile(input_file):
                 print(f"情報: スキップしました（画像ファイルではないようです）: {filename}")
                 skipped_count +=1

        print("\n--- 全ての処理が完了しました ---")
        print(f"成功: {processed_count} 件")
        print(f"エラー: {error_count} 件")
        print(f"スキップ: {skipped_count} 件")
        if processed_count > 0:
             print(f"編集後の画像は {output_dir} に保存されています。")

    elif os.path.isfile(input_path):
        print(f"情報: 単一ファイルを処理します: {input_path}")
        if output_path_arg and os.path.isdir(output_path_arg):
             # 出力先がディレクトリとして指定された場合
             base, ext = os.path.splitext(os.path.basename(input_path))
             output_filename = f"{base}_edited.{args.format.lower()}"
             output_file = os.path.join(output_path_arg, output_filename)
        elif output_path_arg:
             # 出力ファイル名が指定された場合
             output_file = output_path_arg
             # 指定された出力フォーマットと拡張子が異なる場合、拡張子を合わせる
             output_base, output_ext = os.path.splitext(output_file)
             if output_ext.lower() != f".{args.format.lower()}":
                  print(f"警告: 指定された出力ファイル拡張子 {output_ext} とフォーマット {args.format} が異なります。拡張子を .{args.format.lower()} に変更します。")
                  output_file = f"{output_base}.{args.format.lower()}"
        else:
             # 出力パスが省略された場合
             base, ext = os.path.splitext(input_path)
             output_file = f"{base}_edited.{args.format.lower()}"

        process_image(input_path, output_file, args.cascade, output_size, args.format, args.margin)
    else:
        print(f"エラー: 指定された入力パスが見つかりません: {input_path}")
        exit(1)