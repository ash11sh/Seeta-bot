import os,time
import logging
import functools
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from model import styling
from cartoon_gan import cartoon_model
from hair_seg import hair_pred
from io import BytesIO
from skimage.filters import gaussian
from telegram import Update, ChatAction, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext,ConversationHandler


# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)



green = [25,250,32]
yellow = [30,252,249]
orange = [30,108,252]
burgundy = [32,0,128]
ruby =[95,17,224]
blonde = [123,201,227]
DarkRed = [0,0,139]
DarkCyan = [139,139,0]
DarkMagenta = [139,0,139]
Coral = [80,127,255 ]
Violet = [211,0,148]
SeaGreen = [128,205,67]
Peach=[185,218,255]
Gold = [0,215,255]
Chocolate = [19,69,139]

color_dict = {'green':green,'yellow':yellow,'orange':orange,'burgundy':burgundy,'ruby':ruby,
        'blonde':blonde,'DarkRed':DarkRed,'DarkCyan':DarkCyan,'DarkMagenta':DarkMagenta,'Coral':Coral
        ,'Violet':Violet,'SeaGreen':SeaGreen,'Peach':Peach,'Gold':Gold,'Chocolate':Chocolate}



style_dict = {'cubism':'cubism', 'Neo Pop Art':'Neo-Pop_Art','van gogh':'van_gogh',
    'Dali':'Dali','Monet':'Monet','Frida':'Frida','Cezanne':'Cezanne'}


custom_keyboard_0 = [['cartoonizer', 'styling'],['hair color changer']]
markup = ReplyKeyboardMarkup(custom_keyboard_0)

custom_keyboard = [['cubism', 'Neo Pop Art'],['van gogh', 'Dali'],['Monet','Frida','Cezanne']]
reply_markup = ReplyKeyboardMarkup(custom_keyboard)



custom_keyboard_1 = [['yellow','burgundy'],['blonde','Chocolate'],['SeaGreen','Peach']]
markup1 = ReplyKeyboardMarkup(custom_keyboard_1)

def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    # context.bot.send_chat_action(update.message.chat_id, action=ChatAction.TYPING)
    update.message.reply_text('hey send me a photo')
    


def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text('Help!')


def style_commands(update: Update, context: CallbackContext) -> None:
    """Select a style when the command issued."""
    update.message.reply_text('select a style', reply_markup=reply_markup) 

def hair_commands(update: Update, context: CallbackContext) -> None:
    """Select a style when the command issued."""
    update.message.reply_text('select a color', reply_markup=markup1) 


def photo(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    photo_file = update.message.photo[-1].get_file()
    photo_file.download(os.path.join(user.first_name+'.jpg'))

    logger.info("Photo of %s: %s", user.first_name, os.path.join(user.first_name+'.jpg'))
    update.message.reply_text('select a model', reply_markup=markup)



def stylii(update: Update, context: CallbackContext) -> None:
    pathy = update.message.text
    pathy = style_dict[pathy]
    # content_path = 'user_photo.jpg'
    user = update.message.from_user
    filename_suffix='jpg'
    content_path = os.path.join(user.first_name+"." + filename_suffix)
    style_path = os.path.join(pathy + "." + filename_suffix)
    reply_markup = ReplyKeyboardRemove()
    context.bot.send_chat_action(update.message.chat_id, action=ChatAction.TYPING)
    update.message.reply_text('Okay, Now give system a few seconds ⌛⌛ ',reply_markup=reply_markup) 
    context.bot.send_chat_action(update.message.chat_id, action=ChatAction.UPLOAD_PHOTO,timeout=30)
    start_time = time.time()
    bio = styling(content_path,style_path)
    stop_time = time.time()
    if os.path.exists(content_path):
        os.remove(content_path)
    bio.seek(0)
    context.bot.send_photo(update.message.chat_id, photo=bio)
    context.bot.send_chat_action(update.message.chat_id, action=ChatAction.TYPING)
    update.message.reply_text('finished in time: {:.3f} seconds 💃💃🕺🕺'.format((stop_time - start_time)))
    update.message.reply_text('play again with another photo!!')

def cartoon(update: Update, context: CallbackContext) -> None:
    reply_markup = ReplyKeyboardRemove()
    # source_im ='user_photo.jpg'
    user = update.message.from_user
    content_path = os.path.join(user.first_name+'.jpg')
    context.bot.send_chat_action(update.message.chat_id, action=ChatAction.TYPING)
    update.message.reply_text('Okay, Now give system a few seconds ⌛⌛ ',reply_markup=reply_markup) 
    start_time = time.time()
    bio = cartoon_model(content_path)
    stop_time = time.time()
    if os.path.exists(content_path):
        os.remove(content_path)
    bio.seek(0)
    context.bot.send_chat_action(update.message.chat_id, action=ChatAction.UPLOAD_PHOTO)
    context.bot.send_photo(update.message.chat_id, photo=bio)
    context.bot.send_chat_action(update.message.chat_id, action=ChatAction.TYPING)
    update.message.reply_text('finished in time: {:.3f} seconds 💃💃🕺🕺'.format((stop_time - start_time)))
    update.message.reply_text('play again with another photo!!')


def hair(update: Update, context: CallbackContext) -> None:
    hair_color=update.message.text
    # hair_color = hair_dict[hair_color]
    user = update.message.from_user
    content_path = os.path.join(user.first_name+'.jpg')

    hair_color = color_dict[hair_color]
    reply_markup = ReplyKeyboardRemove()
    context.bot.send_chat_action(update.message.chat_id, action=ChatAction.TYPING)
    update.message.reply_text('Okay, Now give system a few seconds ⌛⌛ ',reply_markup=reply_markup) 
    start_time = time.time()
    bio = hair_pred(content_path,hair_color)
    stop_time = time.time()
    if os.path.exists(content_path):
        os.remove(content_path)
    bio.seek(0)
    context.bot.send_chat_action(update.message.chat_id, action=ChatAction.UPLOAD_PHOTO)
    context.bot.send_photo(update.message.chat_id, photo=bio)
    context.bot.send_chat_action(update.message.chat_id, action=ChatAction.TYPING)
    update.message.reply_text('finished in time: {:.3f} seconds 💃💃🕺🕺'.format((stop_time - start_time)))
    update.message.reply_text('Not satisified!? try with well cropped photo!')


def main():
    """Start the bot."""

    TOKEN = " " #place your token

    updater = Updater(TOKEN, use_context=True)

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))

    # on noncommand i.e message - echo the message on Telegram
    dispatcher.add_handler(MessageHandler(Filters.text(['styling']) & ~Filters.command, style_commands))
    dispatcher.add_handler(MessageHandler(Filters.photo & ~Filters.command, photo))
    dispatcher.add_handler(MessageHandler(Filters.text(['cubism', 'Neo Pop Art','van gogh','Dali','Monet','Frida','Cezanne']) & ~Filters.command, stylii))
    dispatcher.add_handler(MessageHandler(Filters.text(['cartoonizer']) & ~Filters.command, cartoon))
    dispatcher.add_handler(MessageHandler(Filters.text(['hair color changer']) & ~Filters.command, hair_commands))
    dispatcher.add_handler(MessageHandler(Filters.text(['green','yellow','orange','burgundy','ruby','blonde','DarkRed','DarkCyan','DarkMagenta','Coral','Violet','SeaGreen','Peach','Gold','Chocolate']) & ~Filters.command, hair))
    # Start the Bot
    # updater.start_polling()

    PORT = int(os.environ.get('PORT', '8413'))

    updater.start_webhook(listen="0.0.0.0",
                      port=PORT,
                      url_path=TOKEN)
    updater.bot.set_webhook("https://your-app.herokuapp.com/" + TOKEN)
    updater.idle()


if __name__ == '__main__':
    main()
