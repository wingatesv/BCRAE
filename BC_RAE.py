# Import Libraries
import tensorflow as tf
import keras

# Residual Block for Resnet18 and RAE18 models
class ResBlock(keras.Model):
  def __init__(self, filters, downsample):
    super().__init__()
    if downsample:
      self.conv1 = keras.layers.Conv2D(filters, 3, 2, padding='same')
      self.shortcut = keras.Sequential([
                keras.layers.Conv2D(filters, 1, 2),
                keras.layers.BatchNormalization()])
    else:
      self.conv1 = keras.layers.Conv2D(filters, 3, 1, padding='same')
      self.shortcut = keras.Sequential()
 
    self.conv2 = keras.layers.Conv2D(filters, 3, 1, padding='same')

  def call(self, input):
    shortcut = self.shortcut(input)

    input = self.conv1(input)
    input = keras.layers.BatchNormalization()(input)
    input = keras.layers.ReLU()(input)

    input = self.conv2(input)
    input = keras.layers.BatchNormalization()(input)
    input = keras.layers.ReLU()(input)

    input = input + shortcut
    return keras.layers.ReLU()(input)
  
  def get_config(self):
    return super().get_config()

  
# Transposed Residual Block for RAE18
class TransResBlock(keras.Model):
  def __init__(self, filters, upsample):
    super().__init__()
    if upsample:
      self.conv1 = keras.layers.Conv2DTranspose(filters, 3, 2, padding='same')
      self.shortcut = keras.Sequential([
                keras.layers.Conv2DTranspose(filters, 1, 2),
                keras.layers.BatchNormalization()])
    else:
      self.conv1 = keras.layers.Conv2DTranspose(filters, 3, 1, padding='same')
      self.shortcut = keras.Sequential()
 
    self.conv2 = keras.layers.Conv2DTranspose(filters, 3, 1, padding='same')

  def call(self, input):
    shortcut = self.shortcut(input)

    input = self.conv1(input)
    input = keras.layers.BatchNormalization()(input)
    input = keras.layers.ReLU()(input)

    input = self.conv2(input)
    input = keras.layers.BatchNormalization()(input)
    input = keras.layers.ReLU()(input)

    input = input + shortcut
    return keras.layers.ReLU()(input)

  def get_config(self):
    return super().get_config()

# RAE18
class RAE18(keras.Model):
    def __init__(self):
        super().__init__()
        self.layer0 = keras.Sequential([
            keras.layers.Conv2D(64, 7, 2, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.MaxPool2D(pool_size=3, strides=1, padding='same')
            ], name='layer0')

        self.layer1 = keras.Sequential([
            ResBlock(64, downsample=True),
            ResBlock(64, downsample=False)
        ], name='layer1')

        self.layer2 = keras.Sequential([
            ResBlock(128, downsample=True),
            ResBlock(128, downsample=False)
        ], name='layer2')

        self.layer3 = keras.Sequential([
            ResBlock(256, downsample=True),
            ResBlock(256, downsample=False)
        ], name='layer3')

        self.layer4 = keras.Sequential([
            ResBlock(512, downsample=True),
            ResBlock(512, downsample=False)
        ], name='layer4')


        self.layer5 = keras.Sequential([
            TransResBlock(512, upsample=True),
            TransResBlock(512, upsample=False)
        ], name='layer5')

        self.layer6 = keras.Sequential([
            TransResBlock(256, upsample=True),
            TransResBlock(256, upsample=False)
        ], name='layer6')

        self.layer7 = keras.Sequential([
            TransResBlock(128, upsample=True),
            TransResBlock(128, upsample=False)
        ], name='layer7')

        self.layer8 = keras.Sequential([
            TransResBlock(64, upsample=True),
            TransResBlock(64, upsample=False)
        ], name='layer8')

        self.layer9 = keras.Sequential([
            keras.layers.Conv2DTranspose(64, 7, 2, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU()
            ], name='layer9')
        
        self.out = keras.layers.Conv2DTranspose(3, 3, 1, padding='same', activation='sigmoid')


    def call(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.layer5(input)
        input = self.layer6(input)
        input = self.layer7(input)
        input = self.layer8(input)
        input = self.layer9(input)
        input = self.out(input)

        return input
    def model(self, input_shape, name):
      x = keras.Input(input_shape)
      return keras.models.Model(x, self.call(x), name=name)
      
      
      
# Bigger Residual Block for Resnet50 and RAE50
class SuperResBlock(keras.Model):
  def __init__(self, filters, stride, downsample):
    super().__init__()
    self.conv1 = keras.layers.Conv2D(filters[0], 1, stride,  padding='valid')
    if downsample:
      self.conv2 = keras.layers.Conv2D(filters[1], 3, 1, padding='same')
      self.shortcut = keras.Sequential([
                keras.layers.Conv2D(filters[2], 1, stride,  padding='valid'),
                keras.layers.BatchNormalization()])
    else:
      self.conv2 = keras.layers.Conv2D(filters[1], 3, 1, padding='same')
      self.shortcut = keras.Sequential()
 
    self.conv3 = keras.layers.Conv2D(filters[2], 1, 1, padding='valid')

  def call(self, input):
    shortcut = self.shortcut(input)

    input = self.conv1(input)
    input = keras.layers.BatchNormalization()(input)
    input = keras.layers.ReLU()(input)

    input = self.conv2(input)
    input = keras.layers.BatchNormalization()(input)
    input = keras.layers.ReLU()(input)

    input = self.conv3(input)
    input = keras.layers.BatchNormalization()(input)
    input = keras.layers.ReLU()(input)

    input = input + shortcut
    return keras.layers.ReLU()(input)
  
  def get_config(self):
    return super().get_config()

  
# Bigger Transposed Residual Block for RAE50
class SuperTransResBlock(keras.Model):
  def __init__(self, filters, stride, upsample):
    super().__init__()
    self.conv1 = keras.layers.Conv2DTranspose(filters[0], 1, stride,  padding='valid')
    if upsample:
      self.conv2 = keras.layers.Conv2DTranspose(filters[1], 3, 1, padding='same')
      self.shortcut = keras.Sequential([
                keras.layers.Conv2DTranspose(filters[2], 1, stride,  padding='valid'),
                keras.layers.BatchNormalization()])
    else:
      self.conv2 = keras.layers.Conv2DTranspose(filters[1], 3, 1, padding='same')
      self.shortcut = keras.Sequential()
 
    self.conv3 = keras.layers.Conv2DTranspose(filters[2], 1, 1, padding='valid')

  def call(self, input):
    shortcut = self.shortcut(input)

    input = self.conv1(input)
    input = keras.layers.BatchNormalization()(input)
    input = keras.layers.ReLU()(input)

    input = self.conv2(input)
    input = keras.layers.BatchNormalization()(input)
    input = keras.layers.ReLU()(input)

    input = self.conv3(input)
    input = keras.layers.BatchNormalization()(input)
    input = keras.layers.ReLU()(input)

    input = input + shortcut
    return keras.layers.ReLU()(input)
  
  def get_config(self):
    return super().get_config()

# RAE50
class RAE50(keras.Model):
    def __init__(self):
        super().__init__()
        self.layer0 = keras.Sequential([
            keras.layers.Conv2D(64, 7, 2, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.MaxPool2D(pool_size=3, strides=1, padding='same')
            ], name='layer0')

        self.layer1 = keras.Sequential([
            SuperResBlock([64,64,256], downsample=True, stride=1),
            SuperResBlock([64,64,256], downsample=False, stride=1),
            SuperResBlock([64,64,256], downsample=False, stride=1)
        ], name='layer1')

        self.layer2 = keras.Sequential([
            SuperResBlock([128, 128, 512], downsample=True, stride=2),
            SuperResBlock([128, 128, 512], downsample=False, stride=1),
            SuperResBlock([128, 128, 512], downsample=False, stride=1),
            SuperResBlock([128, 128, 512], downsample=False, stride=1)
        ], name='layer2')

        self.layer3 = keras.Sequential([
            SuperResBlock([256, 256, 1024], downsample=True, stride=2),
            SuperResBlock([256, 256, 1024], downsample=False, stride=1),
            SuperResBlock([256, 256, 1024], downsample=False, stride=1),
            SuperResBlock([256, 256, 1024], downsample=False, stride=1),
            SuperResBlock([256, 256, 1024], downsample=False, stride=1),
            SuperResBlock([256, 256, 1024], downsample=False, stride=1)
        ], name='layer3')

        self.layer4 = keras.Sequential([
            SuperResBlock([512, 512, 2048], downsample=True, stride=2),
            SuperResBlock([512, 512, 2048], downsample=False, stride=1),
            SuperResBlock([512, 512, 2048], downsample=False, stride=1)
        ], name='layer4')


        self.layer5 = keras.Sequential([
            SuperTransResBlock([2048, 512, 512], upsample=True, stride=2),
            SuperTransResBlock([2048, 512, 512], upsample=False, stride=1),
            SuperTransResBlock([2048, 512, 512], upsample=False, stride=1)
        ], name='layer5')

        self.layer6 = keras.Sequential([
            SuperTransResBlock([1024, 256, 256], upsample=True, stride=2),
            SuperTransResBlock([1024, 256, 256], upsample=False, stride=1),
            SuperTransResBlock([1024, 256, 256], upsample=False, stride=1),
            SuperTransResBlock([1024, 256, 256], upsample=False, stride=1),
            SuperTransResBlock([1024, 256, 256], upsample=False, stride=1),
            SuperTransResBlock([1024, 256, 256], upsample=False, stride=1)
        ], name='layer6')

        self.layer7 = keras.Sequential([
            SuperTransResBlock([512, 128, 128], upsample=True, stride=2),
            SuperTransResBlock([512, 128, 128], upsample=False, stride=1),
            SuperTransResBlock([512, 128, 128], upsample=False, stride=1),
            SuperTransResBlock([512, 128, 128], upsample=False, stride=1)
        ], name='layer7')

        self.layer8 = keras.Sequential([
            SuperTransResBlock([256, 64, 64], upsample=True, stride=1),
            SuperTransResBlock([256, 64, 64], upsample=False, stride=1),
            SuperTransResBlock([256, 64, 64], upsample=False, stride=1)
        ], name='layer8')

        self.layer9 = keras.Sequential([
            keras.layers.Conv2DTranspose(64, 7, 2, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU()
            ], name='layer9')
        
        self.out = keras.layers.Conv2DTranspose(3, 3, 1, padding='same', activation='sigmoid')


    def call(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.layer5(input)
        input = self.layer6(input)
        input = self.layer7(input)
        input = self.layer8(input)
        input = self.layer9(input)
        input = self.out(input)

        return input
    def model(self, input_shape, name):
      x = keras.Input(input_shape)
      return keras.models.Model(x, self.call(x), name=name)
      
      
      
      
# Another version of bigger residual block           
class ResBottleneckBlock(keras.Model):
    def __init__(self, filters, downsample):
        super().__init__()
        self.downsample = downsample
        self.filters = filters
        self.conv1 = keras.layers.Conv2D(filters, 1, 1)
        if downsample:
            self.conv2 = keras.layers.Conv2D(filters, 3, 2, padding='same')
        else:
            self.conv2 = keras.layers.Conv2D(filters, 3, 1, padding='same')
        self.conv3 = keras.layers.Conv2D(filters*4, 1, 1)

    def build(self, input_shape):
        if self.downsample or self.filters * 4 != input_shape[-1]:
            self.shortcut = keras.Sequential([
                keras.layers.Conv2D(
                    self.filters*4, 1, 2 if self.downsample else 1, padding='same'),
                keras.layers.BatchNormalization()
            ])
        else:
            self.shortcut = keras.Sequential()

    def call(self, input):
        shortcut = self.shortcut(input)

        input = self.conv1(input)
        input = keras.layers.BatchNormalization()(input)
        input = keras.layers.ReLU()(input)

        input = self.conv2(input)
        input = keras.layers.BatchNormalization()(input)
        input = keras.layers.ReLU()(input)

        input = self.conv3(input)
        input = keras.layers.BatchNormalization()(input)
        input = keras.layers.ReLU()(input)

        input = input + shortcut
        return keras.layers.ReLU()(input)

# Another version of transposed residual block
class TransResBottleneckBlock(keras.Model):
    def __init__(self, filters, upsample):
        super().__init__()
        self.upsample = upsample
        self.filters = filters
        self.conv1 = keras.layers.Conv2DTranspose(filters * 4, 1, 1)
        if upsample:
            self.conv2 = keras.layers.Conv2DTranspose(filters, 3, 2, padding='same')
        else:
            self.conv2 = keras.layers.Conv2DTranspose(filters, 3, 1, padding='same')
        self.conv3 = keras.layers.Conv2DTranspose(filters, 1, 1)

    def build(self, input_shape):
        if self.upsample or self.filters  != input_shape[-1]:
            self.shortcut = keras.Sequential([
                keras.layers.Conv2DTranspose(
                    self.filters, 1, 2 if self.upsample else 1, padding='same'),
                keras.layers.BatchNormalization()
            ])
        else:
            self.shortcut = keras.Sequential()

    def call(self, input):
        shortcut = self.shortcut(input)

        input = self.conv1(input)
        input = keras.layers.BatchNormalization()(input)
        input = keras.layers.ReLU()(input)

        input = self.conv2(input)
        input = keras.layers.BatchNormalization()(input)
        input = keras.layers.ReLU()(input)

        input = self.conv3(input)
        input = keras.layers.BatchNormalization()(input)
        input = keras.layers.ReLU()(input)

        input = input + shortcut
        return keras.layers.ReLU()(input)

      
#  Another version of RAE50
class RAE50v2(keras.Model):
    def __init__(self):
        super().__init__()
        self.layer0 = keras.Sequential([
            keras.layers.Conv2D(64, 7, 2, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.MaxPool2D(pool_size=3, strides=1, padding='same')
            ], name='layer0')

        self.layer1 = keras.Sequential([
            ResBottleneckBlock(64, downsample=True),
            ResBottleneckBlock(64, downsample=False),
            ResBottleneckBlock(64, downsample=False)
        ], name='layer1')

        self.layer2 = keras.Sequential([
            ResBottleneckBlock(128, downsample=True),
            ResBottleneckBlock(128, downsample=False),
            ResBottleneckBlock(128, downsample=False),
            ResBottleneckBlock(128, downsample=False)
        ], name='layer2')

        self.layer3 = keras.Sequential([
            ResBottleneckBlock(256, downsample=True),
            ResBottleneckBlock(256, downsample=False),
            ResBottleneckBlock(256, downsample=False),
            ResBottleneckBlock(256, downsample=False),
            ResBottleneckBlock(256, downsample=False),
            ResBottleneckBlock(256, downsample=False)
        ], name='layer3')

        self.layer4 = keras.Sequential([
            ResBottleneckBlock(512, downsample=True),
            ResBottleneckBlock(512, downsample=False),
            ResBottleneckBlock(512, downsample=False)
        ], name='layer4')


        self.layer5 = keras.Sequential([
            TransResBottleneckBlock(512, upsample=True),
            TransResBottleneckBlock(512, upsample=False),
            TransResBottleneckBlock(512, upsample=False)
        ], name='layer5')

        self.layer6 = keras.Sequential([
            TransResBottleneckBlock(256, upsample=True),
            TransResBottleneckBlock(256, upsample=False),
            TransResBottleneckBlock(256, upsample=False),
            TransResBottleneckBlock(256, upsample=False),
            TransResBottleneckBlock(256, upsample=False),
            TransResBottleneckBlock(256, upsample=False)
        ], name='layer6')

        self.layer7 = keras.Sequential([
            TransResBottleneckBlock(128, upsample=True),
            TransResBottleneckBlock(128, upsample=False),
            TransResBottleneckBlock(128, upsample=False),
            TransResBottleneckBlock(128, upsample=False)
        ], name='layer7')

        self.layer8 = keras.Sequential([
            TransResBottleneckBlock(64, upsample=True),
            TransResBottleneckBlock(64, upsample=False),
            TransResBottleneckBlock(64, upsample=False)
        ], name='layer8')

        self.layer9 = keras.Sequential([
            keras.layers.Conv2DTranspose(64, 7, 2, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU()
            ], name='layer9')
        
        self.out = keras.layers.Conv2DTranspose(3, 3, 1, padding='same', activation='sigmoid')


    def call(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.layer5(input)
        input = self.layer6(input)
        input = self.layer7(input)
        input = self.layer8(input)
        input = self.layer9(input)
        input = self.out(input)

        return input
    def model(self, input_shape, name):
      x = keras.Input(input_shape)
      return keras.models.Model(x, self.call(x), name=name)
    
    
# Standard Resnet18
class Resnet18(keras.Model):
  def __init__(self, outputs=1):
    super().__init__()
    self.layer0 = keras.Sequential([
      keras.layers.Conv2D(64, 7, 2, padding='same'),
      keras.layers.BatchNormalization(),
      keras.layers.ReLU(),
      keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')
        ], name='layer0')

    self.layer1 = keras.Sequential([
      ResBlock(64, downsample=False),
      ResBlock(64, downsample=False)
        ], name='layer1')

    self.layer2 = keras.Sequential([
      ResBlock(128, downsample=True),
      ResBlock(128, downsample=False)
        ], name='layer2')

    self.layer3 = keras.Sequential([
      ResBlock(256, downsample=True),
      ResBlock(256, downsample=False)
        ], name='layer3')

    self.layer4 = keras.Sequential([
            ResBlock(512, downsample=True),
            ResBlock(512, downsample=False)
        ], name='layer4')

    self.gap = keras.layers.GlobalAveragePooling2D()
    self.fc = keras.layers.Dense(outputs)

  def call(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.gap(input)
        input = self.fc(input)

        return input

  def model(self, input_shape, name):
      x = keras.Input(input_shape)
      return keras.models.Model(x, self.call(x), name=name)
  
  
# Standard Resnet50
class Resnet50(keras.Model):
  def __init__(self, outputs=1):
    super().__init__()
    self.layer0 = keras.Sequential([
      keras.layers.Conv2D(64, 7, 2, padding='same'),
      keras.layers.BatchNormalization(),
      keras.layers.ReLU(),
      keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
        ], name='layer0')

    self.layer1 = keras.Sequential([
      ResBottleneckBlock(64, downsample=False),
      ResBottleneckBlock(64, downsample=False),
      ResBottleneckBlock(64, downsample=False)
        ], name='layer1')

    self.layer2 = keras.Sequential([
      ResBottleneckBlock(128, downsample=True),
      ResBottleneckBlock(128, downsample=False),
      ResBottleneckBlock(128, downsample=False),
      ResBottleneckBlock(128, downsample=False)
        ], name='layer2')

    self.layer3 = keras.Sequential([
      ResBottleneckBlock(256, downsample=True),
      ResBottleneckBlock(256, downsample=False),
      ResBottleneckBlock(256, downsample=False),
      ResBottleneckBlock(256, downsample=False),
      ResBottleneckBlock(256, downsample=False),
      ResBottleneckBlock(256, downsample=False)
        ], name='layer3')

    self.layer4 = keras.Sequential([
            ResBottleneckBlock(512, downsample=True),
            ResBottleneckBlock(512, downsample=False),
            ResBottleneckBlock(512, downsample=False)
        ], name='layer4')

    self.gap = keras.layers.GlobalAveragePooling2D()
    self.fc = keras.layers.Dense(outputs)

  def call(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.gap(input)
        input = self.fc(input)

        return input

  def model(self, input_shape, name):
      x = keras.Input(input_shape)
      return keras.models.Model(x, self.call(x), name=name)
    
