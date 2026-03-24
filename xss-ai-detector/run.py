"""
Entry point cho XSS Detector.

Cách dùng:
  python run.py train svm          # Train SVM
  python run.py train distilbert   # Train DistilBERT
  python run.py train phobert      # Train PhoBERT
  python run.py train all          # Train cả 3
  python run.py serve              # Chạy Flask API (http://localhost:5000)
"""
import sys
import os

BASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, BASE_DIR)


def cmd_train(target: str):
    if target in ('svm', 'all'):
        from training.train_svm import train as train_svm
        print('\n' + '='*55)
        print('▶ Training SVM + TF-IDF')
        print('='*55)
        train_svm()

    if target in ('distilbert', 'all'):
        from training.train_distilbert import train as train_distilbert
        print('\n' + '='*55)
        print('▶ Training DistilBERT')
        print('='*55)
        train_distilbert()

    if target in ('phobert', 'all'):
        from training.train_phobert import train as train_phobert
        print('\n' + '='*55)
        print('▶ Training PhoBERT')
        print('='*55)
        train_phobert()

    if target not in ('svm', 'distilbert', 'phobert', 'all'):
        print(f'Unknown model: {target}')
        print('Options: svm | distilbert | phobert | all')
        sys.exit(1)


def cmd_serve():
    from api.app import app
    from pyngrok import ngrok, conf

    NGROK_TOKEN = '3AtkEdL03tK0yA4jmDxzBOM5wug_597AEFVu4CPjsaAwnopMq'
    conf.get_default().auth_token = NGROK_TOKEN
    ngrok.kill()

    public_url = ngrok.connect(5000)
    print('\n' + '=' * 55)
    print('🚀 XSS Detector API đang chạy!')
    print(f'🌐 Public URL : {public_url}')
    print('📍 Local URL  : http://localhost:5000')
    print('=' * 55)
    app.run(host='0.0.0.0', port=5000, debug=False)


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    cmd = args[0].lower()

    if cmd == 'train':
        target = args[1].lower() if len(args) > 1 else 'all'
        cmd_train(target)
    elif cmd == 'serve':
        cmd_serve()
    else:
        print(f'Unknown command: {cmd}')
        print('Commands: train | serve')
        sys.exit(1)


if __name__ == '__main__':
    main()
