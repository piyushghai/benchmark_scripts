Before running, change the path of MXNet jar in pom.xml

After that, run : 


mvn clean install dependency:copy-dependencies
java -cp target/InferenceAPI-1.0-SNAPSHOT.jar:target/dependency/* sample.PredictorSample