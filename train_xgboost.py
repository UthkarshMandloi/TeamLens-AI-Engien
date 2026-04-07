import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Define the Team
team_members = ["divyansh", "anushka", "love", "uthkarsh"]
task_categories = ["frontend", "backend", "ai_ml", "research", "design", "devops"]

print("📊 Generating Synthetic Workload Dataset...")

# 2. Generate Synthetic Data (Simulating past project tasks)
# Features: task_type, task_complexity, member_current_tasks, member_skill_match
# Label: assigned_member
data = []
for _ in range(500):
    task = np.random.choice(task_categories)
    complexity = np.random.randint(1, 10)

    # Simulate team workloads (0 to 10 tasks currently on their plate)
    workloads = {member: np.random.randint(0, 10) for member in team_members}

    # AI logic determining the "correct" assignment for training
    # It prefers people with lower workloads and specific skills
    best_member = None
    best_score = -999

    for member in team_members:
        # Base score favors lower workloads
        score = -workloads[member] * 2

        # Add skill bonuses
        if member == "divyansh" and task in ["frontend", "design"]: score += 10
        if member == "love" and task in ["backend", "devops"]: score += 10
        if member == "uthkarsh" and task in ["ai_ml", "backend"]: score += 10
        if member == "anushka" and task in ["research", "design"]: score += 10

        if score > best_score:
            best_score = score
            best_member = member

    data.append({
        "task_category": task,
        "complexity": complexity,
        "divyansh_workload": workloads["divyansh"],
        "anushka_workload": workloads["anushka"],
        "love_workload": workloads["love"],
        "uthkarsh_workload": workloads["uthkarsh"],
        "assigned_member": best_member
    })

df = pd.DataFrame(data)

# 3. Preprocess Data
print("⚙️ Preprocessing Data...")
# Convert categorical text into numbers for the XGBoost model
task_encoder = LabelEncoder()
df["task_category_encoded"] = task_encoder.fit_transform(df["task_category"])

member_encoder = LabelEncoder()
df["assigned_member_encoded"] = member_encoder.fit_transform(df["assigned_member"])

X = df[["task_category_encoded", "complexity", "divyansh_workload", "anushka_workload", "love_workload",
        "uthkarsh_workload"]]
y = df["assigned_member_encoded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the XGBoost Model
print("🚀 Training XGBoost Classifier...")
model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=len(team_members),
    eval_metric='mlogloss',
    use_label_encoder=False
)

model.fit(X_train, y_train)

# 5. Evaluate
accuracy = model.score(X_test, y_test)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# 6. Save the Model and Encoders
print("💾 Saving Model to disk...")
model.save_model("task_allocator.json")

# Save category mappings so the FastAPI server knows how to decode the predictions
import json

mapping = {
    "tasks": {label: int(code) for label, code in
              zip(task_encoder.classes_, task_encoder.transform(task_encoder.classes_))},
    "members": {int(code): label for label, code in
                zip(member_encoder.classes_, member_encoder.transform(member_encoder.classes_))}
}
with open("allocator_mapping.json", "w") as f:
    json.dump(mapping, f)

print("🎉 DONE! task_allocator.json is ready to be loaded into FastAPI.")