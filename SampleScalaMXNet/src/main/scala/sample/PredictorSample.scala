package sample

import java.io.File
import scala.io.Source
import javax.imageio.ImageIO
import org.apache.mxnet._
import org.apache.mxnet.infer.{ImageClassifier, Predictor}

object PredictorSample {

  def main(args: Array[String]): Unit = {
    val batchSize = 16
    val modelPathPrefix = "/incubator-mxnet/scala-package/examples/scripts/infer/models/resnet-152/resnet-152"
    val inputImagePath = "/incubator-mxnet/scala-package/examples/scripts/infer/images/dog.jpg"
    val numberOfRuns = 1000
    val context = Array(Context.gpu(0))

    val inputDesc = new DataDesc("data", Shape(batchSize, 3, 224, 224), DType.Float32, Layout.NCHW)

    val predictor = new Predictor(modelPathPrefix, IndexedSeq(inputDesc), context)


    val img = ImageIO.read(new File(inputImagePath))

    val img2 = ImageClassifier.reshapeImage(img, 224, 224)

    val imgND = ImageClassifier.bufferedImageToPixels(img2, Shape(1, 3, 224, 224))
    val imgGPU = imgND.copyTo(context(0))

    if (batchSize == 1) {
      var inferenceTimes: List[Long] = List()
      for (i <- 1 to numberOfRuns) {
          val startTimeSingle = System.nanoTime()
          val output = predictor.predictWithNDArray(IndexedSeq(imgGPU))
          output(0).waitToRead()
    val estimatedTimeSingle = System.nanoTime() - startTimeSingle
          inferenceTimes = estimatedTimeSingle :: inferenceTimes
          println("Inference time at iteration: %d is : %d \n".format(i, estimatedTimeSingle))
    }

      printStatistics(inferenceTimes, "single_inference") 
    }
    else {
      val listND = List.fill(batchSize)(imgGPU)

  val op = NDArray.concatenate(listND)
  op.waitToRead()

      var inferenceTimes2: List[Long] = List()
      val op2 = IndexedSeq(op)

      for (i <- 1 to numberOfRuns) {

          val startTimeSingle = System.nanoTime()
          val output = predictor.predictWithNDArray(op2)
          output(0).waitToRead()
          val estimatedTimeSingle = System.nanoTime() - startTimeSingle
    inferenceTimes2 = estimatedTimeSingle :: inferenceTimes2
          println("Inference time at iteration: %d is : %d \n".format(i, estimatedTimeSingle))
      }

      printStatistics(inferenceTimes2, "batch_inference")
    }
    System.exit(0)

  }

def printResults(predictResultND: Array[Float], modelPath: String): Unit = {



    val bufferedSource = Source.fromFile(modelPath + "/synset.txt")
    val lines = (for (line <- bufferedSource.getLines()) yield line).toList
    bufferedSource.close

    val listT = predictResultND.toList

    val maxIdx = listT.zipWithIndex.maxBy(_._1)._2

    println("Max Class is : " + lines(maxIdx))


  }

  def percentile(p: Int, seq: Seq[Long]): Long = {
    val sorted = seq.sorted
    val k = math.ceil((seq.length - 1) * (p / 100.0)).toInt
    sorted(k)
  }

  def printStatistics(inferenceTimes: List[Long], metricsPrefix: String)  {

    val times: Seq[Long] = inferenceTimes
    val p50 = percentile(50, times) / 1.0e6
    val p99 = percentile(99, times) / 1.0e6
    val p90 = percentile(90, times) / 1.0e6
    val average = times.sum / (times.length * 1.0e6)

    println("\n%s_p99 %f, %s_p90 %f, %s_p50 %f, %s_average %1.2f".format(metricsPrefix,
      p99, metricsPrefix, p90, metricsPrefix, p50, metricsPrefix, average))

  }
}
