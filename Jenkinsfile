pipeline {
  agent any

    stages {

      stage('Checkout'){
        steps {
          checkout scm
        }
      }

      stage('Build') {
        agent {
          docker {
            image 'python:3-alpine' 
          }
        }
        steps {
          sh 'python -m pip install --upgrade pip'
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
