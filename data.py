text = ["my name is JaemoJeong, and I am majoring system AI.", 
        "hello, my name is Jam.", "An signed integer with infinite precision implemented with an \"carrier\" vector of `u32`s.", 
        "Byeonghun Kim is born at 2000.",
        "My key is on my desk. Where is my Key?",
        "Human walks from Africa at first. All of us were from Africa. I have been Europe in this year. It was wonderful. I recommend to you.",
        "The graduation paper and Understanding of Programming HW should be finished until Friday.",
        "Tayeon is in 5th floor. She will come back 30m later.",
        "I ate Eastern Cafeteria Rameon. The trip is planned at Winter vacation to Japan. I will eat many dilicious Ramen.",
        "This Friday is my club's performance day. This week is very busy and I don't want Friday comes :)",
        "Taeyeon is very sick from yesterday. I admire her a lot. Oh, Hyunmin Na likes this sentence!",
        "The sky was painted in hues of orange and pink as the sun dipped below the horizon. In the quiet library, the rustle of pages turning was the only sound to be heard.",
        "cap_hit_ratio = calculate_hit_ratio(cap_tokens, gen_tokens, total_token_length)",
        "Hello, my name is jay em double m oh. I am here to colonize South Korea. from Ethiopia I am, Germany, France anyway many countries are already down!!! "
        ]
dtext = ["의자 위에서 앉아있어요."
        "안녕하세요, 저는 졸업연구를 하고 있습니다. 꽤 오래 걸리네요 ㅎㅎ;",
        "김병훈은 스물 다섯 살이다.",
        "한국말 데이터를 만들고 있는데 쓸 말이 없네요. 추천 부탁드립니다.",
        "허교수님은 최고시다. 이 랩실에 꼭 들어가고 싶다.",
        "카이스트 포스텍 서울대 연새대 고려대 레츠고",
        "요즘들어 인공지능과 딥러닝 기술의 발전은 사람들의 일상생활에 엄청난 변화를 가져오고 있습니다.",
        "학교에 다닐 때 가장 기억에 남는 일은 친구들과 함께 동아리 활동을 하며 많은 추억을 쌓았던 것이었어요.",
        "여행을 다녀오고 나면 항상 느끼는 점이 하나 있어요. 새로운 사람을 만나고, 그곳의 문화를 체험하며 얻게 되는 교훈은 책에서 배우는 것과는 전혀 다르다는 것이죠.",
        "이번 연구에서는 데이터 분석을 통해 얻은 통찰을 기반으로, 소비자 행동을 예측할 수 있는 모델을 개발하고자 합니다.",
        "어느 조용한 도서관에서 책을 읽던 중, 갑자기 창문 밖에서 들려온 빗소리가 저를 현실로 데려왔습니다.",
        "대한민국은 한반도 남쪽에 위치한 나라로, 세계적으로도 IT 강국으로 알려져 있습니다.",

        "내가 처음으로 자전거를 배웠던 날은 여전히 선명히 기억납니다. 엄마가 뒤에서 자전거를 잡아주며 달려가던 모습이 정말 귀여웠는데, 제가 균형을 잡고 혼자 달릴 수 있게 되자 마치 날아가는 기분이 들었어요.",]
prompt = {"G1": "Summarize this section.", "G2": 'Highlight the critical points from this section.', 
          "G": 'Summarize the critical points highlighted in this section.', 
          "Q": 'Considering the following question, summarize the critical points highlighted in this section. Question: {question}',
          "U": 'Unrelated (U) The sky is blue. The sun is yellow. Here we go. There and back again.',
          "P": '\n\n\n...',
}
ratioes = [2,3,4] # for T
C_arr = [4, 6, 8] # for C
M_arr = [8, 10, 20 , 30] # for M