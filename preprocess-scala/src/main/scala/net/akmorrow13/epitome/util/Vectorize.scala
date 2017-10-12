package net.akmorrow13.epitome.util

import java.io._

import net.akmorrow13.epitome.EpitomeArgs
import org.apache.spark.SparkContext
import org.bdgenomics.adam.rdd.GenericGenomicRDD
import org.bdgenomics.adam.rdd.read.AlignmentRecordRDD
import org.bdgenomics.adam.rdd.feature.{FeatureRDD, CoverageRDD}
import org.bdgenomics.formats.avro.{AlignmentRecord, Feature}
import org.apache.spark.rdd.RDD
import org.bdgenomics.adam.models.{Coverage, ReferenceRegion}
import breeze.linalg.DenseVector
import org.bdgenomics.adam.rdd.ADAMContext._

case class Vectorizer(@transient sc: SparkContext, @transient conf: EpitomeArgs) {

  conf.validate()

  /**
   * Takes reads and features, and returns an RDD of label and feature pairs
   *
   * @return RDD of (vector of labels, vector of features)
   */
  def partitionAndFeaturize(): RDD[(DenseVector[Int], DenseVector[Int])] = {

    val featurePathLabelArray = conf.getFeaturePathLabelsArray()


    val (reads: AlignmentRecordRDD, features: FeatureRDD)  = {
      val reads = sc.loadAlignments(conf.readsPath)

      // copy over because conf is transient

      // load all feature files and set feature type to TF name
      var features: FeatureRDD = conf.getFeaturePathsArray().zipWithIndex
        .map(r => {
          sc.loadFeatures(r._1).transform(rdd => rdd.map(x => {
          Feature.newBuilder(x)
            .setFeatureType(featurePathLabelArray(r._2))
            .build()
      }))
      }
      ).reduce(_ union _)

      features = features.replaceSequences(reads.sequences)

      if (conf.partitions > 0) {
        (reads.transform(r => r.repartition(conf.partitions)), features.transform(r => r.repartition(conf.partitions)))
      } else {
        (reads, features)
      }

    }


    // Step 1: Get candidate regions (ATAC called peaks)

    // TODO: START replace with MACS2 caller
    val coverage = reads.toCoverage()

    val filteredCoverage: CoverageRDD = coverage.transform(rdd => {
      rdd.filter(r => r.count > 2) // todo remove hardcode
        .map(r => new Coverage(r.contigName, r.start - (r.start % 1000), r.end + 1000 - (r.end % 1000), r.count)) // TODO combine if overlapping/next to eachother
        .keyBy(r => ReferenceRegion(r))
        .reduceByKey((a,b) => a)
        .map(r => r._2)
    })
    // TODO END replace with MACS2 caller

    require(filteredCoverage.rdd.count>0, "filteredCoverageRDD empty")

    // Step 2: Join ATAC peaks and ChIP-seq peaks Join(atac, chipseq)
    // use this line instead of the above when getting an error for unequal number of partitions
    val joinedPeaks = filteredCoverage.rightOuterShuffleRegionJoinAndGroupByLeft(features)
      .rdd.filter(_._1.isDefined)
      .map(r => {
        val labels = r._2.map(_.getFeatureType).toArray.distinct
        val vector = DenseVector.fill(featurePathLabelArray.length){0}

        // Create vector of TF binding sites
        featurePathLabelArray.zipWithIndex.foreach(label => {
          if (labels.contains(label._1))
            vector(label._2) = 1
        })
        (r._1.get, vector)
      })

  // cache reads for join
  reads.rdd.cache()
  reads.rdd.count()

  // region function for (coverage, feature) tuples, used for GenomicRDD
  def regionFn(r: (Coverage, DenseVector[Int])): Seq[ReferenceRegion] = Seq(ReferenceRegion(r._1))

  val peakGenomicRDD = GenericGenomicRDD(joinedPeaks, filteredCoverage.sequences, regionFn)

  // cache peaks for join
  peakGenomicRDD.rdd.cache()
  println("peakGenomicRDD count",peakGenomicRDD.rdd.count())

  // Step 3: join dense vector of features and reads
  val labelsAndAlignments: RDD[(ReferenceRegion, DenseVector[Int], Iterable[AlignmentRecord])] =
    peakGenomicRDD.rightOuterShuffleRegionJoinAndGroupByLeft(reads).rdd
    .filter(r => r._1.isDefined)
    .map(r => (ReferenceRegion(r._1.get._1), r._1.get._2, r._2))

  // Step 4: Featurize Reads into DenseVector of cutsite counts
  vectorizeCutSites(labelsAndAlignments)

  }

  /**
   * Featurizes Iterable[AlignmentRecords] by ReferenceRegion to DenseVector of cut sites
   *
   * @param featurizedCuts RDD[(Region, label vector, reads to featurize for this datapoint)]
   * @return RDD[(Labels, Features)]
   */
  private def vectorizeCutSites(featurizedCuts: RDD[(ReferenceRegion, DenseVector[Int], Iterable[AlignmentRecord])]): RDD[(DenseVector[Int], DenseVector[Int])] = {

    // get window size
    val windowSize = conf.windowSize

    val featurized: RDD[(DenseVector[Int], DenseVector[Int])] =
      featurizedCuts.map(window => {
        val region = window._1

        val positions = DenseVector.zeros[Int](windowSize)

        val reads: Map[ReferenceRegion, Int] = window._3
          .flatMap(r => Iterable(ReferenceRegion(r.contigName, r.start, r.start + 1),
            ReferenceRegion(r.contigName, r.end, r.end+1)))
          .map(r => (r, 1))
          .groupBy(_._1)
          .map(r => (r._1, r._2.map(_._2).toArray.sum))

        reads.filter(cut => {
          val index = (cut._1.start - region.start).toInt
          index >= 0 && index < windowSize
        }).foreach(cut => {
          positions((cut._1.start - region.start).toInt) += cut._2
        })

        // positions should be size of window
        (window._2, positions)
      })

    featurized
  }

  /**
   * Saves features to local file
   *
   * @param RDD of labels, features
   * @param filepath to save data to
   */
  def saveValuesLocally(rdd: RDD[(DenseVector[Int], DenseVector[Int])], filepath: String) = {

    val collected= rdd.collect

    val file = new PrintWriter(new File(s"${filepath}/featuresAndLabels.txt"))

    val header = s"#${conf.featurePathLabels}\n"

    file.write(header)

    collected.toList.foreach(r => {
      file.write(r._1.toArray.mkString(",") + ";" + r._2.toArray.mkString(",") + "\n")
    })

    file.close()

  }

}
