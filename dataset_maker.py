# 학습에 필요한 데이터셋을 만드는 코드
# 약 200 시간의 피아노 연주 미디 파일로 이루어진 MAESTRO V3.0.0 데이터를 사용 (출처: https://magenta.tensorflow.org/datasets/maestro)
# 데이터는 (피아노 건반을 누른 타이밍, 누르고 있었던 시간, 피아노 음(pitch), 건반을 누른 세기, 곡의 조성) 으로 구성

from music21 import *
import os
import json
import pretty_midi


key_dict = {"D_major": 0,
       "A_major": 1,
       "A_minor": 2,
       "C_major": 3,
       "C#_minor": 4,
       "A-_major": 5,
       "B_major": 6,
       "F_minor": 7,
       "F#_minor": 8,
       "G_major": 9,
       "B_minor": 10,
       "E-_major": 11,
       "F#_major": 12,
       "C#_major": 13,
       "G#_minor": 14,
       "C_minor": 15,
       "E_major": 16,
       "B-_minor": 17,
       "B-_major": 18,
       "E-_minor": 19,
       "E_minor": 20,
       "F_major": 21,
       "D_minor": 22,
       "G_minor": 23}


def make_dataset():
    midi_dir_path = "midi_dataset/maestro/test/"

    midi_list = os.listdir(midi_dir_path)

    dataset = []

    print('start')

    file_num = 0
    for midi_name in midi_list:
        file_num += 1
        print(file_num, midi_dir_path + midi_name)

        # 조성 분석을 위해 music21 라이브러리로 미디 파일 로드
        midi_stream = converter.parse(midi_dir_path + midi_name)
        key = midi_stream.analyze('key')
        midi_key = key.tonic.name + '_' + key.mode
        midi_key = key_dict[midi_key]

        # 데이터셋을 만들기 위해 pretty_midi 라이브러리로 미디 파일 로드
        midi = pretty_midi.PrettyMIDI(midi_dir_path + midi_name)
        
        # 미디 파일에 사용된 악기가 하나인지 검사
        assert len(midi.instruments) == 1
        
        # 피아노 노트 정보로 이루어진 데이터 만들기
        notes = [
            (
                # 피아노 건반을 누른 타이밍
                float(n.start),
                # 건반을 누르고 있었던 시간
                float(n.end) - float(n.start),
                # 피아노 음
                int(n.pitch - 21),
                # 건반을 누른 세기
                int(n.velocity),
                # 곡의 조성
                int(midi_key)
            )
            for n in midi.instruments[0].notes
        ]

        # 건반을 누른 타이밍 순서로 정렬
        notes = sorted(notes, key=lambda n: (n[0], n[2]))
        assert all(
            [
                all(
                    [
                        # Start times should be non-negative
                        n[0] >= 0,
                        # Note durations should be strictly positive
                        n[1] > 0,
                        # Key index should be in range of the piano
                        0 <= n[2] and n[2] < 88,
                        # Velocity should be valid
                        1 <= n[3] and n[3] < 128,
                    ]
                )
                for n in notes
            ]
        )

        dataset.append(notes)

    # json 형태로 데이터셋 저장
    dataset_json = {
        "train": dataset[0:int(len(dataset) * 0.7)],
        "validation": dataset[int(len(dataset) * 0.7):int(len(dataset) * 0.85)],
        "test": dataset[int(len(dataset) * 0.85):len(dataset)]
    }

    with open('dataset/new_dataset.json', 'w') as f:
        json.dump(dataset_json, f)


if __name__ == "__main__":
    make_dataset()

