# 학습된 모델을 활용한 피아노 연주를 위한 코드
# 피아노 연주에는 25-key midi 키보드를 사용하였으며, 연주에 흰 건반 10개, 장/단조 조절에 추가적인 건반 1개가 사용

import random
from ghost_pianist import GhostPianistAutoencoder, CFG, IntegerQuantizer
import time
import tkinter
import pygame.midi
import socket
import rtmidi
import torch


# 피아노 연주, 버튼, 조성을 모델의 입력으로 넣어 현재 연주될 피아노 음을 생성하고 재생하는 함수
def play_note(k, t, b, m, velocity):
    # 피아노 음 생성
    hat_k = model.dec(k, t, b, m)
    new_key = torch.max(hat_k, dim=2).indices.tolist()[0][-1]
    
    # 생성된 음 재생
    player.note_on(new_key + 21, velocity=velocity)

    return new_key


# 연주를 위한 업데이트 함수 1ms 마다 실행
def update():
    global prev_note_time, k_inp, b_inp, t_inp, m_inp, key_signature

    message = midi_in.getMessage()
    
    # 미디 입력이 있다면 실행
    if message:
        note = message.getNoteNumber()
        velocity = message.getVelocity()

        # 피아노 건반이 눌렸다면 실행
        if message.isNoteOn():
            # 눌린 피아노 건반이 72라면 장/단조 전환
            if note == 72:
                if key_signature == 22:
                    # 유니티 화면에서 장/단조 표시를 위한 정보 전달
                    sock.sendto(str.encode('Happy'), serverAddressPort)
                    key_signature = 3   # C_Major
                elif key_signature == 3:
                    # 유니티 화면에서 장/단조 표시를 위한 정보 전달
                    sock.sendto(str.encode('Sad'), serverAddressPort)
                    key_signature = 22  # D_minor
            # 눌린 피아노 건반이 key_btn_mapper 안에 있다면 모델을 사용해 피아노 음 생성 후 재생
            elif note in key_btn_mapper:
                # 유니티 화면에서 피아노 애니메이션을 위한 정보 전달
                sock.sendto(str.encode('on ' + str(note)), serverAddressPort)

                if velocity < 40:
                    velocity = 40

                # 현재 눌린 피아노 건반 정보로 모델 입력 생성
                b_inp = torch.cat((b_inp, torch.tensor([[key_btn_mapper[note]]]).long().to("cuda:0")), dim=1)
                b_inp = IQ.discrete_to_real(b_inp)
                t_inp = torch.cat((t_inp, torch.tensor([[time.time() - prev_note_time]]).float().to("cuda:0")), dim=1)
                m_inp = torch.cat((m_inp, torch.tensor([[key_signature]]).long().to("cuda:0")), dim=1)

                # 생성된 입력으로 피아노 음을 생성하고 재생
                new_key = play_note(k_inp, t_inp, b_inp, m_inp, velocity)
                k_inp = torch.cat((k_inp, torch.tensor([[new_key]]).long().to("cuda:0")), dim=1)

                # 건반을 누른 세기에 비례하여 추가적인 피아노 음 생성 및 재생
                if velocity <= 40:
                    prob = 0
                elif velocity >= 100:
                    prob = 1
                else:
                    prob = (velocity-40) / 60
                rand = random.random()
                if rand < prob:
                    b_inp = torch.cat((b_inp, torch.tensor([[0]]).long().to("cuda:0")), dim=1)
                    b_inp = torch.cat((b_inp, torch.tensor([[10]]).float().to("cuda:0")), dim=1)
                    t_inp = torch.cat((t_inp, torch.tensor([[0.001]]).float().to("cuda:0")), dim=1)
                    m_inp = torch.cat((m_inp, torch.tensor([[key_signature]]).long().to("cuda:0")), dim=1)

                    new_key = play_note(k_inp, t_inp, b_inp, m_inp, velocity)
                    k_inp = torch.cat((k_inp, torch.tensor([[new_key]]).long().to("cuda:0")), dim=1)

                prev_note_time = time.time()

        # 피아노 건반이 때졌다면 실행
        if message.isNoteOff():
            # 유니티 화면에서 피아노 애니메이션을 위한 정보 전달
            sock.sendto(str.encode('off ' + str(note)), serverAddressPort)

    root.after(int(1000/fps), update)


if __name__ == "__main__":
    # 피아노 건반 입력을 0~9 사이의 버튼 입력으로 맵핑하는 딕셔너리
    key_btn_mapper = {48: 0, 50: 0, 52: 1, 53: 1, 55: 2, 57: 3, 59: 4, 60: 5, 62: 6, 64: 7, 65: 8, 67: 9, 69: 9}

    # Create model
    model = GhostPianistAutoencoder(CFG)
    model.to("cuda:0")
    model_pt = torch.load('./piano_genie/model.pt', map_location=torch.device("cuda"))
    model.load_state_dict(model_pt)

    # 유니티와 통신을 위한 UDP 소켓 생성
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    serverAddressPort = ("192.168.0.101", 7777)

    # MIDI 출력을 위한 초기화
    pygame.midi.init()
    player = pygame.midi.Output(0)
    player.set_instrument(0)  # 피아노 소리를 사용하기 위해 채널 0을 선택
    # MIDI 입력 장치 생성
    midi_in = rtmidi.RtMidiIn()
    # 사용 가능한 MIDI 입력 장치 목록 가져오기
    midi_in.openPort(0)

    # 모델 입력 및 관련 변수들 초기화
    IQ = IntegerQuantizer(CFG["num_buttons"])
    k_inp = torch.tensor([[88]]).long().to("cuda:0")
    b_inp = torch.tensor([[]]).long().to("cuda:0")
    t_inp = torch.tensor([[]]).long().to("cuda:0")
    m_inp = torch.tensor([[]]).long().to("cuda:0")
    key_signature = 3  # 3: C_Major, 22: D_minor, 장조를 위해 C_Major를 단조를 위해 D_minor를 사용
    prev_note_time = time.time()

    # 업데이트 함수를 1ms 마다 실행
    fps = 1000
    root = tkinter.Tk()
    root.after(int(1000/fps), update)
    root.mainloop()

