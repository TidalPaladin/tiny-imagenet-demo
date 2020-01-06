pipeline {
  agent any

    stages {

      stage('Checkout'){
        steps {
          checkout scm
        }
      }

      stage('Test') {
        steps {
          sh 'pytest --cov=tin --cov-report=xml'
        }
      }

      stage('Build') {
        steps {
          echo 'docker-compose build'
        }
      }

    }
}
