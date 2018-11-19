package sample

import java.io.File

import javax.imageio.ImageIO
import org.apache.mxnet._
import org.apache.mxnet.infer.{ImageClassifier, Predictor}
import org.apache.mxnet.io.NDArrayIter
import org.apache.mxnet.module.Module
import sample.PredictorSample.{printResults, printStatistics}

object ModuleExample {

  def main(args: Array[String]): Unit = {

    val modelPathPrefix = "/tmp/resnet152/resnet-152"
    val inputImagePath = "/tmp/resnet50_ssd/images/dog.jpg"
    val numberOfRuns = 1000

    val context = Array(Context.cpu())

    val inputDesc = new DataDesc("data", Shape(1, 3, 224, 224), DType.Float32, Layout.NCHW)

    //val predictor = new Predictor(modelPathPrefix, IndexedSeq(inputDesc), context)

    val ndarry = NDArray.ones(Shape(1, 3, 224, 224))

    val img = ImageIO.read(new File(inputImagePath))

    val img2 = ImageClassifier.reshapeImage(img, 224, 224)

    val imgND = ImageClassifier.bufferedImageToPixels(img2, Shape(1, 3, 224, 224))

    val module = Module.loadCheckpoint(modelPathPrefix, 0 , contexts = context)

    module.bind(dataShapes = IndexedSeq(inputDesc))


    val iter = new NDArrayIter(IndexedSeq(imgND), dataBatchSize = 1)



    var inferenceTimes: List[Long] = List()
    for (i <- 1 to numberOfRuns) {
      val startTimeSingle = System.nanoTime()
      val output = module.predict(iter)
      NDArray.waitall()
      val estimatedTimeSingle = System.nanoTime() - startTimeSingle

      inferenceTimes = estimatedTimeSingle :: inferenceTimes
      println("Inference time at iteration: %d is : %d \n".format(i, estimatedTimeSingle))
      printResults(output(0).toArray, "/tmp/resnet152/")
    }

    printStatistics(inferenceTimes, "single_inference")

  }
}
