package mxnet;

import org.apache.mxnet.infer.javaapi.Predictor;
import org.apache.mxnet.javaapi.*;
import org.apache.mxnet.javaapi.Shape;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class PredictorExample {

    public static void main(String[] args) throws IOException {

        String modelPathPrefix = "/tmp/resnet152/resnet-152";

        String inputImagePath = "/tmp/resnet152/dog.jpg";

        boolean useBatch = false;

        int batchSize = 1;

        int numberOfRuns = 1000;

        java.util.List<Context> context = getContext();

        Shape inputShape = new Shape(new int[] {batchSize, 3, 224, 224});

        java.util.List<DataDesc> inputDescriptors = new ArrayList<DataDesc>();
        inputDescriptors.add(new DataDesc("data", inputShape, DType.Float32(), "NCHW"));

        BufferedImage img = loadIamgeFromFile(inputImagePath);

        img = reshapeImage(img, 224, 224);
	
        NDArray arr = new NDArray(imagePreprocess(img), new Shape(new int[] {1, 3, 224, 224}), new Context("gpu", 0));

        NDArray gpuArray = arr.copyTo(context.get(0));

        Predictor predictor = new Predictor(modelPathPrefix, inputDescriptors, context,0);


        if (!useBatch) {
            List<NDArray> l = new ArrayList<>();
            l.add(gpuArray);

            long[] times = new long[numberOfRuns];

            for (int i = 0; i < numberOfRuns; i++) {
                long currTime = System.nanoTime();
                List<NDArray> res = predictor.predictWithNDArray(l);
                res.get(0).waitToRead();
                //res.get(0);
                times[i] = System.nanoTime() - currTime;

                System.out.println("Inference time at iteration: " + i + " is : " + (times[i] / 1.0e6) + "\n");
                System.out.println(printMaximumClass(res.get(0).toArray(), modelPathPrefix));
            }

            printStatistics(times, "single_inference");
        }

        else {

            NDArray[] array = new NDArray[batchSize];

            for (int i = 0; i < batchSize; i++) {
                array[i] = gpuArray;
            }

            NDArray[] arr2 = NDArray.concat(array, array.length, 0, null);

            List<NDArray> l2 = new ArrayList<>();
            l2.add(arr2[0]);

            arr2[0].waitToRead();

            long[] times2 = new long[numberOfRuns];

            for (int i = 0; i < numberOfRuns; i++) {
                long currTime = System.nanoTime();
                List<NDArray> res = predictor.predictWithNDArray(l2);
                res.get(0).waitToRead();
                times2[i] = System.nanoTime() - currTime;
                System.out.println("Inference time at iteration: " + i + " is : " + (times2[i] / 1.0e6) + "\n");
            }

            printStatistics(times2, "batch_inference");
        }

        return;

    }

    private static List<List<String>> generateBatches(String imageFolderPath, int batchSize) {

        File directory = new File(imageFolderPath);

        List<List<String>> res = new ArrayList<>();
        List<String> batch = new ArrayList<>();

        for (File f : directory.listFiles()) {

            batch.add(f.getAbsolutePath());

            if (batch.size() == batchSize) {
                res.add(batch);
                batch = new ArrayList<String>();
            }

        }

        if (batch.size() > 0) {
            res.add(batch);
        }

        return res;

    }

    private static String printMaximumClass(float[] probabilities) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader("/tmp/resnet50/synset.txt"));
        ArrayList<String> list = new ArrayList<String>();
        String line = reader.readLine();

        while (line != null){
            list.add(line);
            line = reader.readLine();
        }
        reader.close();

        int maxIdx = 0;
        for (int i = 1;i<probabilities.length;i++) {
            if (probabilities[i] > probabilities[maxIdx]) {
                maxIdx = i;
            }
        }

        return "Probability : " + probabilities[maxIdx] + " Class : " + list.get(maxIdx) ;
    }

    private static java.util.List<Context> getContext() {
        java.util.List<Context> ctx = new ArrayList<>();
        ctx.add(new Context("gpu", 0));

        return ctx;
    }

    private static BufferedImage loadIamgeFromFile(String inputImagePath) throws IOException {
        return ImageIO.read(new File(inputImagePath));
    }

    public static BufferedImage reshapeImage(BufferedImage buf, int newWidth, int newHeight) {
        BufferedImage resizedImage = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resizedImage.createGraphics();
        g.drawImage(buf, 0, 0, newWidth, newHeight, null);
        g.dispose();
        return resizedImage;
    }

    public static float[] imagePreprocess(BufferedImage buf) {
        // Get height and width of the image
        int w = buf.getWidth();
        int h = buf.getHeight();

        // get an array of integer pixels in the default RGB color mode
        int[] pixels = buf.getRGB(0, 0, w, h, null, 0, w);

        // 3 times height and width for R,G,B channels
        float[] result = new float[3 * h * w];

        int row = 0;
        // copy pixels to array vertically
        while (row < h) {
            int col = 0;
            // copy pixels to array horizontally
            while (col < w) {
                int rgb = pixels[row * w + col];
                // getting red color
                result[0 * h * w + row * w + col] = (rgb >> 16) & 0xFF;
                // getting green color
                result[1 * h * w + row * w + col] = (rgb >> 8) & 0xFF;
                // getting blue color
                result[2 * h * w + row * w + col] = rgb & 0xFF;
                col += 1;
            }
            row += 1;
        }
        buf.flush();
        return result;
    }

    private static long percentile(int p, long[] seq) {
        Arrays.sort(seq);
        int k = (int) Math.ceil((seq.length - 1) * (p / 100.0));
        return seq[k];
    }

    private static void printStatistics(long[] inferenceTimesRaw, String metricsPrefix)  {
        long[] inferenceTimes = inferenceTimesRaw;
        // remove head and tail
        if (inferenceTimes.length > 2) {
            inferenceTimes = Arrays.copyOfRange(inferenceTimesRaw,
                    1, inferenceTimesRaw.length - 1);
        }
        double p50 = percentile(50, inferenceTimes) / 1.0e6;
        double p99 = percentile(99, inferenceTimes) / 1.0e6;
        double p90 = percentile(90, inferenceTimes) / 1.0e6;
        long sum = 0;
        for (long time: inferenceTimes) sum += time;
        double average = sum / (inferenceTimes.length * 1.0e6);

        System.out.println(
                String.format("\n%s_p99 %fms\n%s_p90 %fms\n%s_p50 %fms\n%s_average %1.2fms",
                        metricsPrefix, p99, metricsPrefix, p90,
                        metricsPrefix, p50, metricsPrefix, average)
        );

    }

 /**
     * Helper class to print the maximum prediction result
     * @param probabilities The float array of probability
     * @param modelPathPrefix model Path needs to load the synset.txt
     */
    private static String printMaximumClass(float[] probabilities,
                                            String modelPathPrefix) throws IOException {
        String synsetFilePath = modelPathPrefix.substring(0,
                1 + modelPathPrefix.lastIndexOf(File.separator)) + "/synset.txt";
        BufferedReader reader = new BufferedReader(new FileReader(synsetFilePath));
        ArrayList<String> list = new ArrayList<>();
        String line = reader.readLine();

        while (line != null){
            list.add(line);
            line = reader.readLine();
        }
        reader.close();

        int maxIdx = 0;
        for (int i = 1;i<probabilities.length;i++) {
            if (probabilities[i] > probabilities[maxIdx]) {
                maxIdx = i;
            }
        }

        return "Probability : " + probabilities[maxIdx] + " Class : " + list.get(maxIdx) ;
    }
}
