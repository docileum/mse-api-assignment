# Malawi Stock Exchange API Assignment

## Overview
Build a comprehensive REST API to serve Malawi Stock Exchange (MSE) historical data. You'll process PDF reports spanning 2017-present, handle varying data formats, store data in PostgreSQL, and create API endpoints for data access.

## Timeline (4 Weeks)

### Week 1: Data Extraction & Processing (Sept 22-28)
**Focus**: Master PDF-to-CSV pipeline
- [ ] Analyze PDF report formats from 2017-present
- [ ] Update `mse_pdf2csv.py` to handle all report formats
- [ ] Process all historical reports to extract daily prices data into CSV files
- [ ] Generate clean, standardized CSV files

**Deliverable**: Working extraction notebook + processed CSV files. 

### Week 2: Database Design & Implementation (Sept 29 - Oct 5)
**Focus**: PostgreSQL setup and data management
- [ ] Design optimal database schema for historical data
- [ ] Create tables and relationships
- [ ] Implement data loading mechanisms

**Deliverable**: Database with all historical data loaded

### Week 3: API Development (Oct 6-12)
**Focus**: Build REST API endpoints
- [ ] Implement all 3-5 required API endpoints
- [ ] Add input validation using Pydantic models
- [ ] Add query parameters for filtering

**Deliverable**: Complete working API with all endpoints

### Week 4: Documentation, Testing & Polish (Oct 13-19)
**Focus**: Professional finishing touches
- [ ] Write comprehensive API documentation
- [ ] Prepare screenshots and demo materials
- **Optional**: Deploy API using ngrok for live evaluation

**Deliverable**: Production-ready API with full documentation and demo materials

**Final Submission Deadline: October 20, 2025**

## Data Source
**PDF Reports**: Download all MSE historical reports from [Google Drive](https://drive.google.com/drive/folders/1Bg7MiCViXxpPIB3jfpkClUM1HiLVrAEn?usp=sharing)

**Important**: Download all PDF reports and place them in the `data/raw_pdfs/` folder before starting the assignment.

## Setup
1. Clone this repository
2. Download PDF reports from the Google Drive link above
3. Place all PDFs in the `data/raw_pdfs/` folder
4. Create other necessary folders such as `csv_files`
5. Install dependencies: `pip install -r requirements.txt`
6. Copy `.env.example` to `.env` and configure database connection

## Assignment Tasks

### Task 1: PDF Data Extraction (25%)
**Objective**: Modify the existing `mse_pdf2csv.py` script to process all MSE reports from 2017 to present. Each report contains daily stock prices in tabular format - your task is to extract these complete daily price tables and convert them to CSV format. Reference the [example CSV files](https://drive.google.com/drive/folders/1-XTEDGa982HmXqpk92w5Du3oIHTvCMIq?usp=sharing) from our class exercises to understand the expected output structure and format. 

**Current Status**: The existing script in `src/utils/mse_pdf2csv.py` works for reports up to 2021. Your task is to modify/extend it to handle all reports through the present.

**Challenges to Address**:
- **Varying number of tickers** across different time periods
- **Format changes** in report tables post-2021
- **Inconsistent table structures** between older and newer reports
- **Data quality issues** and missing values. 

**Deliverables**:
- [ ] Updated `mse_pdf2csv.py` script that handles all report formats (2017-present)
- [ ] Error handling for problematic PDFs
- [ ] Data validation and cleaning
- [ ] Daily CSV files in a folder withing the data folder with standardized format
- [ ] Recommended: merged csv file for use in next step


### Task 2: Database Management (15%)
**Objective**: Set up and populate a PostgreSQL database with your extracted stock price data. You may build upon the database created during our class exercise. Use the same database schema and ticker information sources as outlined in our class exercise - refer to this [class document](https://docs.google.com/document/d/1w3N9B7hw41iS5cheyZYfnVUVdqLGiOmn-94eDJ4jRCQ/edit?usp=sharing) for the complete database structure and ticker table specifications.

**Requirements**:
- [ ] Create a database, you can call it `mse_db` or any other name
- [ ] Add two tables, one for `tickers` (as we did in class exercise) and another for `daily prices`
- [ ] Make sure you allow missing values in some columns as necessary 
- [ ] Handle duplicate records and data conflicts
- [ ] Load/import data into the database tables


### Task 3: API Development (20%)
**Objective**: Complete the REST API endpoints

**Required Endpoints**: *[Details will be updated later]*
1. `GET /endpoint1` - [Description to be provided]
2. `GET /endpoint2` - [Description to be provided]
3. `GET /endpoint3` - [Description to be provided]
4. `GET /endpoint4` - [Description to be provided]
5. `GET /endpoint5` - [Description to be provided]

**API Requirements**:
- [ ] Proper error handling and HTTP status codes
- [ ] Input validation using Pydantic models
- [ ] Query parameters for filtering as required
- [ ] Comprehensive API documentation


### Task 4: Project Demo & Documentation (40%)
**Objective**:
Showcase your functional API using detailed visual documentation. Create a `screenshots` folder inside the `docs` directory where instructors can review your demonstration materials as outlined in the requirements below.

**Required Screenshots** (submit in `screenshots/` folder):
- [ ] **Startup**: Terminal showing `uvivorn main:app --reload` running successfully
- [ ] **API Home**: Browser showing API running at `http://localhost:8000`
- [ ] **Interactive Docs**: FastAPI docs at `http://localhost:8000/docs`
- [ ] **Endpoint Testing**: Each of the 3-5 endpoints returning sample data
- [ ] **Database Connection**: Evidence of successful database connection (pgAdmin/terminal)

**Screenshot Requirements**:
- High-quality, clear images showing full browser/terminal windows
- Include timestamps or other proof of recency
- Name files descriptively: `01_api_startup.png`, `02_interactive_docs.png`, etc.


## Bonus Tasks (Optional-Additional Points)

### Live API Demo via ngrok (10% points)
**For Advanced Students**: Deploy your API using ngrok for live instructor testing

**Requirements**:
- [ ] Install and configure ngrok
- [ ] Deploy API with public URL
- [ ] Submit ngrok URL for real-time evaluation
- [ ] Schedule brief demo session with instructor
- [ ] Ensure API remains accessible during evaluation period

**Submission**: Include ngrok URL and availability schedule in your final submission

## Evaluation Criteria & Grading

| Component | Points | Evaluation Method |
|-----------|--------|-------------------|
| **PDF Processing** | 25% | Code review + output quality |
| **Database Design** | 15% | Schema design + data loading success |
| **API Implementation** | 20% | Code quality + endpoint functionality |
| **Project Documentation** | 40% | **Screenshot demonstration** |
| **Bonus: Live Demo** | +10% | **Real-time ngrok testing** |

## Critical Success Criteria
**To receive full credit, your project MUST:**
1. Process all PDF reports from 2017-present successfully
2. Load data into PostgreSQL database with proper schema
3. Implement all 3-5 API endpoints with proper validation
4. **Provide clear screenshots demonstrating full functionality**


## Submission Guidelines
1. **Code**: Push all code to your GitHub repository. This is what will be used as your submission. Instructors will clone your repository on the final submission day.
2. **Screenshots**: Include `screenshots/` folder with all required images
3. **Documentation**: Update README with your specific setup instructions
4. **Database**: Include SQL schema and sample data files
5. **Optional**: Submit ngrok URL for bonus evaluation

**Final Submission Deadline: October 20, 2025**

## Technical Requirements
- Use FastAPI for API development
- PostgreSQL for data storage
- Pandas and other packages for data processing
- Clean, documented code
- Clear visual documentation via screenshots

---