 # ok now do the same with TitaNet-LID Rep-TDNN whus VoxLingua107 ECAPA-TDNN CommonLanguage ECAPA

import nemo.collections.asr as nemo_asr
speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")

emb = speaker_model.get_embedding("/media/jhowe/BACKUPBOY/fake_wavs/openai_fake_conversation_1747206400811_wvk62e.wav")