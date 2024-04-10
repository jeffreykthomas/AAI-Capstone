# TherapAI
## AAI Capstone Project - Team 4
### Team Members: 
- [Maimuna Bashir](https://github.com/maymoonah-bash)
- [Cesar Lucero](https://github.com/CexarLuxzero92)
- [Jeffrey Thomas](https://github.com/jeffreykthomas)
- [Ruben Velarde](https://github.com/RV249)

### Project Description:
TherapAI is an AI Language model developed to aid people during difficult times in their life and through anxiety and depression episodes. Powered by Natural Language Processing and Sentiment Analysis, TherapAI is an aiding tool for empathetic emotional support.

### Project Objectives:
- Develop a chatbot that can provide emotional support to users.
- Implement a sentiment analysis model to detect the user's emotional state.
- Develop a user-friendly interface for the chatbot.

### Project Structure:
- **Data**: The data used was from multiple sources. For pretraining, we used [openwebtext](). For dialogue fine-tuning, we used [Empathetic Dialogues]() and [Daily Dialoges](). And then for mental health dialogue fine-tuning, we used [Mental Health Dialogues](), ...
- **Data Preprocessing**: The data was cleaned and preprocessed to remove any unnecessary information.
- **Model Development**: The chat model was developed using the LlAMa architecture and fine-tuned on the mental health dialogues.
- **User Interface**: The user interface was developed using React-Native.
- **Deployment**: The chatbot was deployed using Firebase.

### Project Replication:
- Clone the repository:
```bash
git clone
```
- Install the required packages:
```bash
pip install -r requirements.txt
```
- Prepare the training data:
```bash
python prepare_data.py
```
- Run the base training script:
```bash
python train.py
```
- Run the fine-tuning script:
```bash
python train.py --dataset mental_health --pretrained_model_path /path/to/pretrained/model \
--learning_rate 5e-5 --num_global_steps 5000
```

- Run the distillation training:
```bash
python train.py --dataset openwebtext --load_pretrained True --distill_training \
--pretrained_path /path/to/weights --output_dir /output/path --micro_batch_size 4 \
--accumulation_steps=64 --dropout 0.1 --warmup_steps 1000 --num_global_steps 5000  \
--learning_rate 5e-5 --lr_decay_iters 5000 --grad_clip 1.0 --weight_decay 0.1  \
--eval_steps 50 --eval_iters 100 --log_interval 10 --use_galore
```

### Project Demo:
- [TherapAI Chatbot](https://therapai-chatbot.herokuapp.com/)