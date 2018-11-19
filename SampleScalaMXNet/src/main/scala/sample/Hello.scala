package sample

import org.apache.mxnet._

object Hello {
  def main(args: Array[String]): Unit = {
    println("hello World")
    val arr = NDArray.ones(2, 3)
    println(arr.shape)
  }
}
