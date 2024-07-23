# Movie Recommender (ML Project)

This project is a movie recommendation system that leverages multiple collaborative filtering techniques, including user-based and item-based collaborative filtering, as well as matrix factorisation using Singular Value Decomposition (SVD). The system combines these methods to provide hybrid recommendations to users.

## Project Objective

As an aspiring data scientist eager to enter the world of machine learning, I chose to develop a movie recommendation system for my project. This decision was driven by my interest in understanding complex algorithms and my passion for leveraging data to enhance user experiences. Through this project, I am gaining hands-on experience in data preprocessing, collaborative filtering, Singular Value Decomposition (SVD), and model evaluation. By the end of this project, I will have not only built a functional recommendation system but also deepened my understanding of various recommendation techniques. This experience is a significant step toward my goal of becoming a skilled data scientist, capable of transforming theoretical knowledge into practical applications that can drive business insights and innovation.

## Features

- **User-based Collaborative Filtering:** Recommends movies based on the similarity between users.
- **Item-based Collaborative Filtering:** Recommends movies based on the similarity between items (movies).
- **Singular Value Decomposition (SVD):** Matrix factorization technique to predict missing entries in the user-item matrix.
- **Hybrid Recommendation:** Combines the strengths of user-based, item-based, and SVD methods to provide robust recommendations.
- **Model Evaluation:** Includes functions to evaluate the performance of the recommendation system using Mean Squared Error (MSE), precision, recall, and F1-score.

## Dataset

The dataset used in this project is the MovieLens dataset, which contains user ratings for various movies.

## Installation

### Prerequisites

- Python 3.6 or higher
- pip package manager

### Dependencies

Install the required packages using the following command:

```sh
pip install -r requirements.txt
```

**Here are the dependencies listed in the requirements.txt file:**

- pandas
- numpy
- scikit-learn
- requests
- flask
- flask-cors
- seaborn
- matplotlib

## Usage
### Running the Recommendation System
Clone the Repository:
```sh
git clone https://github.com/naeem0512/MovieRecommender.git
cd MovieRecommender
```

Run the Python Script:
```sh
python MovieRecommender.py
```

## Flask API
A simple Flask API is provided to get movie recommendations for a user.

#### Start the Flask Server:
```sh
python MovieRecommender.py
```

**Make API Requests:** 
You can make GET requests to the /recommend endpoint to get movie recommendations for a user. For example:
```sh
curl "http://127.0.0.1:5000/recommend?user_id=1&num_recommendations=5"
```

## File Descriptions
- **MovieRecommender.py:** Main script containing the implementation of the recommendation system and the Flask API. You can run this script directly to see the recommendation system in action.
- **MovieRecommender.ipynb:** Jupyter Notebook containing the implementation and step-by-step explanation of the recommendation system. This notebook is ideal for those who prefer a more detailed exploration of the code with explanations and insights into how each component works.
- **requirements.txt:** List of dependencies required to run the project. Ensure these packages are installed to successfully run the scripts.

Feel free to explore both the .py and .ipynb files to gain a complete understanding of how the recommendation system functions and how it has been developed. Each file offers a unique perspective and depth to the project, catering to different learning styles and technical approaches.

## Conclusion
This project has been incredibly rewarding, laying a solid foundation for my understanding and implementation of various recommendation techniques. By experimenting with user-based, item-based, and hybrid approaches, I have gained valuable insights into the intricacies of building and evaluating recommendation systems. Each phase of the project challenged me to think critically and creatively, enhancing my problem-solving skills and deepening my appreciation for data-driven decision-making. The experience has not only bolstered my technical expertise but also reinforced my passion for data science as a means to create impactful solutions.

Feel free to explore, modify, and enhance the project. Contributions are welcome!

## Contributing
Developing the **MovieRecommender** Flask app has been one of the most challenging yet rewarding parts of this project. I recognise that there might be areas of the app that are not up to the standard I aspire to, as I am still learning and perfecting my craft. This is why I deeply appreciate any contributions that can help enhance and refine the app. Whether it's improving the codebase, adding new features, or identifying and fixing bugs, your expertise and insights are invaluable.

If you’re interested in contributing, please feel free to fork the repository, make your changes, and submit a pull request. I am eager to learn from your contributions and collaborate on advancing this project further. For more detailed information on how to contribute, please see the **CONTRIBUTING.md** file in the repository.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
