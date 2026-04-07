# **TeamLens AI Engine** 

The intelligent backend microservice powering the **TeamLens** collaborative platform. Built with Python and FastAPI, this engine handles natural language processing, predictive task allocation, and meeting summarization to eliminate unequal contribution in team projects.

## **Features**

1. **AI Task Extraction (LLaMA-3):** Analyzes chat messages in real-time to automatically identify and extract actionable tasks, assignees, and deadlines.  
2. **Smart Allocation (XGBoost):** Evaluates team members' current workloads and skill sets to mathematically predict the optimal person for a newly created task.  
3. **Meeting Intelligence:** Parses meeting transcripts to generate dynamic debriefs containing key decisions, action items, and overall sentiment.

## **Tech Stack**

* **Framework:** FastAPI / Uvicorn  
* **Machine Learning:** XGBoost, Scikit-Learn, Pandas  
* **NLP Model:** Meta LLaMA-3 8B (Parameter-Efficient Fine-Tuning via LoRA/Unsloth)  
* **Environment:** Python 3.10+

## **Local Setup & Installation**

If you want to run the AI Engine locally on your machine, follow these steps:

### **1\. Clone the Repository**
```bash
git clone https://github.com/UthkarshMandloi/TeamLens_Client.git  
cd TeamLens-AI-Engien 
```

### **2\. Create a Virtual Environment**

It is highly recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv  
# On Windows:  
.\venv\Scripts\activate  
# On Mac/Linux:  
source venv/bin/activate 
```

### **3\. Install Dependencies**
```bash
pip install -r requirements.txt
```
### **4\. Run the API Server**
```bash
uvicorn main:app --reload --port 8000
```

*The API will be live at http://localhost:8000.*

*Interactive API Documentation (Swagger UI) is available at http://localhost:8000/docs.*

##  **Model Training & Generation Guides**

This repository includes the scripts necessary to generate datasets and train the AI models from scratch.

### **A. Smart Task Allocator (XGBoost)**

The XGBoost model predicts task assignments based on workload and skills.

1. Run the training script:  
   python train\_xgboost.py

2. This will generate a synthetic dataset, train the model, and output two files: task\_allocator.json and allocator\_mapping.json.  
3. The FastAPI server will automatically load these files on startup.

### **B. NLP Task Extraction (LLaMA-3)**

Due to file size constraints (100MB+), the fine-tuned LLaMA-3 LoRA adapters are stored externally.

**To use the pre-trained model:**

1. Download the model .zip file from our Google Drive:
    <p align="center">
  <i><span style="font-size:30px;">[👉 GOOGLE DRIVE LINK HERE](https://drive.google.com/drive/folders/1tbDfxcdj3yF9pOwIarEALsnImWYS_qvL?usp=sharing)
  </p>
3. Extract the folder into the root directory of this repository.

**To train the model yourself:**

1. Run the dataset generation script locally to create the JSONL training data:  
   python generate\_dataset.py

2. Open our fully guided Google Colab Notebook:
   <p align="center">
  <i><span style="font-size:30px;">[👉 GOOGLE COLAB LINK HERE](https://colab.research.google.com/drive/1AqKONt9j8UpJpcOK0S539a5dFJA37u2i?usp=sharing)
   </p>
4. Upload the task\_extraction\_dataset.jsonl file to the Colab environment.  
5. Follow the instructions in the notebook to fine-tune LLaMA-3 using Unsloth and export your own model weights.

##  **API Endpoints Reference**

### **POST /api/ai/smart-allocate**

Calculates the best team member for a specific task.

**Payload:**

```bash
{  
  "task_description": "Build the new database schema for the backend",  
  "available_members": ["divyansh", "anushka", "love", "uthkarsh"]  
}
```

### **POST /api/ai/summarize-meeting**

Generates a structured debrief from an array of chat messages.

**Payload:**

```bash
\[  
  {  
    "user_id": "uthkarsh",  
    "message": "i have completed the AI model fine tuning",  
    "timestamp": "2024-10-27T10:00:00Z"  
  }  
\]
```

### **POST /api/ai/extract-task**

*(Currently mocked for testing, requires LLaMA-3 integration for production)*

Extracts task data from raw chat input.

**Payload:**

```bash
{  
  "user_id": "love",  
  "message": "/assign @divyansh to fix the navbar padding",  
  "timestamp": "2024-10-27T10:00:00Z"  
} 
```
