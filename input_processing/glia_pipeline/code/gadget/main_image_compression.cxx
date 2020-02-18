#include "util/image_io.hxx"
#include "util/text_cmd.hxx"
using namespace glia;

bool operation (std::string const& outputImageFile,
                std::string const& inputImageFile,
                bool write16, bool compress)
{
  switch (readImageInfo(inputImageFile)->GetComponentType()) {
    case itk::ImageIOBase::IOComponentType::UCHAR:
      {
        typedef itk::Image<unsigned char, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::CHAR:
      {
        typedef itk::Image<char, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::USHORT:
      {
        typedef itk::Image<unsigned short, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::SHORT:
      {
        typedef itk::Image<short, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::UINT:
      {
        typedef itk::Image<unsigned int, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        if (write16) {
          castWriteImage<UInt16Image<DIMENSION>>
              (outputImageFile, image, compress);
        }
        else { writeImage(outputImageFile, image, compress); }
        break;
      }
    case itk::ImageIOBase::IOComponentType::INT:
      {
        typedef itk::Image<int, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::ULONG:
      {
        typedef itk::Image<unsigned long, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        if (write16) {
          castWriteImage<UInt16Image<DIMENSION>>
              (outputImageFile, image, compress);
        }
        else { writeImage(outputImageFile, image, compress); }
        break;
      }
    case itk::ImageIOBase::IOComponentType::LONG:
      {
        typedef itk::Image<long, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::FLOAT:
      {
        typedef itk::Image<float, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        writeImage(outputImageFile, image, compress);
        break;
      }
    case itk::ImageIOBase::IOComponentType::DOUBLE:
      {
        typedef itk::Image<double, DIMENSION> Image;
        auto image = readImage<Image>(inputImageFile);
        writeImage(outputImageFile, image, compress);
        break;
      }
    default: perr("Error: unsupported image pixel type...");
  }
  return true;
}


int main (int argc, char* argv[])
{
  std::string inputImageFile, outputImageFile;
  bool write16 = false,  compress;
  bpo::options_description opts("Usage");
  opts.add_options()
      ("help", "Print usage info")
      ("inputImage,i", bpo::value<std::string>(&inputImageFile)->required(),
       "Input label image file name")
      ("write16,u", bpo::value<bool>(&write16),
       "Whether to write (uint/ulong) to uint16 images [default: false]")
      ("compress,z", bpo::value<bool>(&compress)->required(),
       "Whether to compress (otherwise uncompress) output image file")
      ("outputImage,o",
       bpo::value<std::string>(&outputImageFile)->required(),
       "Output label image file name");
  return
      parse(argc, argv, opts) &&
      operation(outputImageFile, inputImageFile, write16, compress)?
      EXIT_SUCCESS: EXIT_FAILURE;
}
