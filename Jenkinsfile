pipeline {
  agent any

    stages {

      stage('Checkout'){
        steps {
          checkout scm
        }
      }

      stage('Build') {
        steps {
          sh 'pip install --user -r requirements.txt'
        }
      }

      stage('Test') {
        steps {
          sh 'pytest --cov=tin --cov-report=xml'
        }
      }


    }
}
