package Machine_Learning_with_Spark

import java.io.PrintWriter
import java.net.ServerSocket

import scala.util.Random

/**
  * A producer application that generates random "product events",
  * up to 5 pre second, and sends them over a network connection
  */

object StreamingProducer {
    def main(args: Array[String]): Unit = {
        val random = new Random()

        //maximum number of events per second
        val MaxEvents = 6

        //Read the list of possible names
        val namesResource = this.getClass.getResourceAsStream("/Users/wangqi/Desktop/names.csv")
        val names = scala.io.Source.fromInputStream(namesResource)
          .getLines()
          .toList
          .head
          .split(",")
          .toSeq
        //generate a sequence of possible products
        val products = Seq(
            "iPhone Cover" -> 9.99,
            "Headphones" -> 5.49,
            "Samsung Galaxy Cover" -> 8.95,
            "iPad Cover" -> 7.49
        )

        /** generate a number of random product events */
        def generateProductEvents(n: Int) = {
            (1 to n).map { i =>
                val (product, price) =
                    products(random.nextInt(products.size))
                val user = random.shuffle(names).head
                (user, product, price)
            }
        }

        /**
          * Finally, we will create a network socket and set our producer to listen on this socket.
          * as soon as a connection is made (which will come from our consumer streaming application),
          * the producer will start generating random events at a random rate between 0 and 5 per second
           */
        //create a newwork producer
        val listener = new ServerSocket(9999)
        println("Listening on port: 9999")

        while (true) {
            val socket = listener.accept()
            new Thread() {
                override def run = {
                    println("Got client connected from: " +
                    socket.getInetAddress)
                    val out = new PrintWriter(socket.getOutputStream(), true)
                    while (true) {
                        Thread.sleep(1000)
                        val num = random.nextInt(MaxEvents)
                        val productEvents = generateProductEvents(num)
                        productEvents.foreach {
                            event =>
                                out.write(event.productIterator.mkString(","))
                                out.write("\n")
                        }
                        out.flush()
                        println(s"Created $num events...")
                    }
                    socket.close()
                }
            }.start()
        }
    }
}
