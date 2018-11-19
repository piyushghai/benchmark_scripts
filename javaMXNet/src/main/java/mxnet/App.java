package mxnet;

import org.apache.mxnet.javaapi.Context;
import org.apache.mxnet.javaapi.NDArray;

public class App 
{
    public static void main( String[] args )
    {
        NDArray nd = NDArray.ones(Context.cpu(), new int[] {10, 20});
        System.out.println( "Testing MXNet by generating a 10x20 NDArray" );
        System.out.println("Shape of NDArray is : " + nd.shape());
    }
}
