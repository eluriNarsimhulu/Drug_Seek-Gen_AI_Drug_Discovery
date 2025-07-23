# 🧬 Drug_Seek-Gen_AI_Drug_Discovery

![Drug Discovery](https://img.shields.io/badge/AI-Drug%20Discovery-blue) ![GenAI](https://img.shields.io/badge/GenAI-Lung%20Disease-green) ![Status](https://img.shields.io/badge/Status-Active-success)

## 📋 Project Overview

The advent of GenAI has brought forth the potential to revolutionize the accuracy and efficiency of lung disease diagnosis, with a specific focus on CT images. One of the most significant opportunities of software development to contribute to the pharmaceutical industry is to enhance drug discovery and imaging diagnosis.

This comprehensive web application leverages **Generative AI** for:
- 🫁 **Lung Disease Diagnosis** on CT images
- 💊 **Drug Design Tools** to identify active drug molecules
- 🔬 **Protein-Ligand Affinity Prediction**
- 🧪 **Protein Structure Prediction**
- ⚛️ **Autodocking** using ML/GenAI

Custom software solutions, such as virtual screening and molecular modeling software, contribute enormously to the drug discovery process.

## 🎥 Project Demo

📹 **[Watch Project Explanation Video](https://drive.google.com/file/d/1Y0fV6HUXdmhyOjNYYlReN3i6-4g3H6UB/view?usp=sharing)**

*This video explains the key points and functionality of the project.*

## 🏗️ Project Structure

```
Drug_Seek-Gen_AI_Drug_Discovery/
├── client/                 # Frontend React application
│   ├── src/
│   │   └── components/
│   │       └── lung/       # Lung disease diagnosis components
│   │           └── app.py  # Python Flask server for AI processing
│   ├── package.json
│   └── ...
├── server/                 # Backend Node.js server
│   ├── package.json
│   └── ...
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables (create this)
└── README.md
```

## ⚙️ Installation & Setup

### Prerequisites
- Node.js (v14 or higher)
- Python (v3.8 or higher)
- MongoDB Atlas account
- Gmail account (for email services)
- Twilio account (for SMS services)

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/Drug_Seek-Gen_AI_Drug_Discovery.git
cd Drug_Seek-Gen_AI_Drug_Discovery
```

### Step 2: Install Dependencies

#### Frontend Dependencies
```bash
cd client
npm install
```

#### Backend Dependencies
```bash
cd ../server
npm install
```

### Step 3: Python Environment Setup
```bash
# Navigate to root directory
cd ..

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### Step 4: Environment Configuration

Create a `.env` file in the root directory with the following configuration:

```env
# Database Configuration
DB="your url"
# JWT Configuration
JWTPRIVATEKEY=mySuperSecretPrivateKey123!
SALT=10

# Server Configuration
BASE_URL=http://localhost:8080

# Email Configuration
HOST=smtp.gmail.com
SERVICE=gmail
EMAIL_PORT=587
SECURE=false
USER=drugseek.med@gmail.com
PASS=qbgn rytt uugd awcz

# Twilio Configuration (SMS Services)
TWILIO_ACCOUNT_SID=AXXX3efbea761c8b14815846
TWILIO_AUTH_TOKEN=eXXXXXbeed40156b73d13c9246930a61
TWILIO_PHONE_NUMBER=+18XXX467841
```

> ⚠️ **Important**: Replace placeholder values (XXXX) with your actual credentials.

## 🚀 Running the Application

### Method 1: Start All Services

1. **Start Frontend (Client)**
```bash
cd client
npm start
```
*Frontend will run on `http://localhost:3000`*

2. **Start Backend (Server)**
```bash
cd server
npm start
```
*Backend will run on `http://localhost:8080`*

3. **Start Python Flask Server (AI Processing)**
```bash
cd client/src/components/lung
python app.py
```
*Flask server will handle AI/ML processing*

> 📝 **Note**: If you encounter path errors in the Flask server, adjust the file paths in `app.py` according to your system configuration.

### Method 2: Development Mode
For development, you can run all services concurrently using terminal tabs or your preferred method.

## 🔧 Configuration Notes

### Database Setup
- Ensure your MongoDB Atlas cluster is running
- Update the `DB` connection string with your actual password
- Whitelist your IP address in MongoDB Atlas

### Email Service
- The project uses Gmail SMTP for email notifications
- Ensure "Less secure app access" is enabled or use App Passwords

### Twilio Integration
- Used for SMS notifications and communications
- Obtain credentials from your Twilio Console

## 🌟 Features

- **🤖 AI-Powered Diagnosis**: Advanced GenAI models for lung disease detection
- **💊 Drug Discovery**: Molecular modeling and virtual screening
- **🔬 Protein Analysis**: Structure prediction and ligand binding analysis
- **📊 Real-time Results**: Instant processing and visualization
- **📱 Responsive Design**: Works across all devices
- **🔐 Secure Authentication**: JWT-based user authentication
- **📧 Notifications**: Email and SMS alert system

## 🛠️ Technologies Used

- **Frontend**: React.js, HTML5, CSS3, JavaScript
- **Backend**: Node.js, Express.js
- **AI/ML**: Python, Flask, TensorFlow/PyTorch
- **Database**: MongoDB Atlas
- **Authentication**: JWT
- **Communication**: Twilio (SMS), Gmail (Email)

## 📝 API Endpoints

The application provides RESTful APIs for:
- User authentication and authorization
- CT image upload and processing
- Drug molecule analysis
- Protein structure prediction
- Results retrieval and management

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**
- GitHub: [@your-username](https://github.com/your-username)
- Email: your.email@example.com

## 🙏 Acknowledgments

- Thanks to the open-source community for various libraries and tools
- Special recognition to AI/ML researchers advancing drug discovery
- Healthcare professionals providing domain expertise

---

### 🆘 Troubleshooting

**Common Issues:**

1. **Port conflicts**: Ensure ports 3000, 8080, and Flask port are available
2. **Python path errors**: Check and adjust file paths in `app.py`
3. **Database connection**: Verify MongoDB Atlas credentials and IP whitelist
4. **Email service**: Confirm Gmail settings and app passwords

**Need Help?** 
- Check the [project video](https://drive.google.com/file/d/1Y0fV6HUXdmhyOjNYYlReN3i6-4g3H6UB/view?usp=sharing) for detailed explanation
- Open an issue on GitHub
- Contact the development team

---

*Built with ❤️ for advancing healthcare through AI*
