# Ghost Pianist
피아노를 배워보지 않았고 음악 이론 지식이 없는 초보자들을 도와서 즉흥 연주를 할 수 있게 만들어 주는 AI Assistant
10개의 버튼으로 피아노 연주가 가능하며 이를 위해 오토인코더를 학습 시켰음
Piano Genie 논문(https://magenta.tensorflow.org/pianogenie)의 오토인코더 방식을 참고 및 수정 하였으며 MAESTRO V3.0.0 데이터셋(https://magenta.tensorflow.org/datasets/maestro)을 활용하였음음

## 스크립트 설명
ghost_pianist.py: 오토인코더 모델 스크립트
dataset_maker: MAESTRO V3.0.0 데이터셋을 활용하여 데이터셋을 만드는 스크립트
piano.py: 학습된 모델과 MIDI 피아노를 활용하여 실제 연주를 하는 스크립트
hand_tracking: VR 연주를 위해 실제 손 움직임을 추적하고 정보를 Unity로 전달하기 위한 스크립트


