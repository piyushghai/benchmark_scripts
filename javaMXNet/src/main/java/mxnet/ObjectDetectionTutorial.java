package mxnet;

import org.apache.mxnet.infer.javaapi.ObjectDetector;
import org.apache.mxnet.infer.javaapi.ObjectDetectorOutput;
import org.apache.mxnet.infer.javaapi.Predictor;
import org.apache.mxnet.javaapi.Context;
import org.apache.mxnet.javaapi.DType;
import org.apache.mxnet.javaapi.DataDesc;
import org.apache.mxnet.javaapi.Shape;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

public class ObjectDetectionTutorial {


    public static void main(String[] args) throws IOException {

        String modelPathPrefix = "/tmp/resnet50/resnet-50";

        String inputImagePath = "/tmp/resnet50_ssd/images/dog.jpg";

        List<Context> context = getContext();

        Shape inputShape = new Shape(new int[] {1, 3, 224, 224});

        List<DataDesc> inputDescriptors = new ArrayList<DataDesc>();
        inputDescriptors.add(new DataDesc("data", inputShape, DType.Float32(), "NCHW"));

        return;

    }

    private static List<Context> getContext() {
        List<Context> ctx = new ArrayList<>();
        ctx.add(Context.cpu());

        return ctx;
    }

}
